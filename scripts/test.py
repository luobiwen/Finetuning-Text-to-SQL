import json
import re
import torch
from tqdm import tqdm
from transformers import GenerationConfig
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import config

def parse_prompts_file(prompts_file):
    """
    解析prompts文件，提取每个prompt块
    """
    with open(prompts_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式分割不同的prompt块
    pattern = r'=== Prompt \d+ \(SQL ID: (sql_\d+)\) ===(.*?)' + re.escape('=' * 80)
    matches = re.findall(pattern, content, re.DOTALL)
    
    prompts = []
    for sql_id, prompt_content in matches:
        prompts.append({
            'sql_id': sql_id.strip(),
            'content': prompt_content.strip()
        })
    
    return prompts

def extract_sql_from_text(text):
    """
    从生成的文本中提取SQL语句
    使用多种策略来确保提取到正确的SQL
    """
    # 策略1: 查找SQL关键字
    sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'ALTER', 'DROP']
    
    # 查找第一个SQL关键字的位置
    first_keyword_pos = -1
    first_keyword = None
    
    for keyword in sql_keywords:
        pos = text.upper().find(keyword)
        if pos != -1 and (first_keyword_pos == -1 or pos < first_keyword_pos):
            first_keyword_pos = pos
            first_keyword = keyword
    
    if first_keyword_pos != -1:
        # 从第一个SQL关键字开始截取
        sql_candidate = text[first_keyword_pos:]
        
        # 尝试找到SQL的结束位置（分号或明显的非SQL内容）
        semicolon_pos = sql_candidate.find(';')
        if semicolon_pos != -1:
            sql = sql_candidate[:semicolon_pos + 1].strip()
        else:
            # 如果没有分号，找到第一个明显的非SQL行
            lines = sql_candidate.split('\n')
            sql_lines = []
            for line in lines:
                line_upper = line.upper().strip()
                # 如果行包含SQL关键字或是空行或是注释，继续
                if any(keyword in line_upper for keyword in sql_keywords) or not line.strip() or line.strip().startswith('--'):
                    sql_lines.append(line)
                else:
                    # 遇到非SQL内容，停止
                    break
            sql = '\n'.join(sql_lines).strip()
        
        return sql
    
    # 策略2: 查找代码块
    code_block_pattern = r'```(?:\w+)?\s*(.*?)```'
    code_matches = re.findall(code_block_pattern, text, re.DOTALL)
    if code_matches:
        return code_matches[0].strip()
    
    # 策略3: 如果以上都失败，返回前200个字符（假设SQL在开头）
    return text[:200].strip()

def batch_generate_sql(prompts_file, model, tokenizer, output_file=None, batch_size=8, device=None):
    """
    使用批量生成加速SQL生成过程
    """
    # 设置设备
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 确保模型在正确的设备上
    model = model.to(device)
    model.eval()  # 设置为评估模式
    
    # 优化生成配置
    generation_config = GenerationConfig(
        max_new_tokens=512,  # 增加最大生成长度
        do_sample=False,
        temperature=0.1,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=1,
        early_stopping=True,
    )
    
    # 解析prompts文件
    prompts = parse_prompts_file(prompts_file)
    print(f"找到 {len(prompts)} 个prompts，使用批量大小: {batch_size}")
    
    results = []
    
    # 分批处理prompts
    for i in tqdm(range(0, len(prompts), batch_size), desc="生成SQL"):
        batch_prompts = prompts[i:i+batch_size]
        # 使用更明确的指令
        batch_texts = [
            p['content'] + "\n\n请只生成SQL查询语句，不要包含任何解释、注释或其他文本。直接输出SQL语句。" 
            for p in batch_prompts
        ]
        batch_ids = [p['sql_id'] for p in batch_prompts]
        
        try:
            # 批量编码输入
            inputs = tokenizer(
                batch_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=2048
            )
            # 确保所有输入都在同一设备上
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # 批量生成SQL
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                    output_scores=False
                )
            
            # 解码输出
            generated_texts = tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
            
            # 提取SQL部分
            for j, (sql_id, prompt_text, generated_text) in enumerate(zip(batch_ids, batch_texts, generated_texts)):
                # 获取生成部分
                generated_part = generated_text[len(prompt_text):].strip()
                
                # 使用新的提取方法
                sql = extract_sql_from_text(generated_part)
                
                results.append({
                    'sql_id': sql_id,
                    'sql': sql
                })
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU内存不足，减少批量大小到 {max(1, batch_size//2)}")
                torch.cuda.empty_cache()
                return batch_generate_sql(prompts_file, model, tokenizer, output_file, max(1, batch_size//2), device)
            else:
                print(f"运行时错误 (批次 {i//batch_size + 1}): {e}")
                # 回退到逐条生成
                for prompt in batch_prompts:
                    try:
                        enhanced_prompt = prompt['content'] + "\n\n请只生成SQL查询语句，不要包含任何解释、注释或其他文本。直接输出SQL语句。"
                        sql = generate_single_sql(enhanced_prompt, model, tokenizer, device)
                        
                        sql = extract_sql_from_text(sql)
                        
                        results.append({
                            'sql_id': prompt['sql_id'],
                            'sql': sql
                        })
                    except Exception as e2:
                        print(f"生成SQL时出错 (SQL ID: {prompt['sql_id']}): {e2}")
                        results.append({
                            'sql_id': prompt['sql_id'],
                            'sql': f"ERROR: {str(e2)}"
                        })
        except Exception as e:
            print(f"批量生成SQL时出错 (批次 {i//batch_size + 1}): {e}")
            # 回退到逐条生成
            for prompt in batch_prompts:
                try:
                    enhanced_prompt = prompt['content'] + "\n\n请只生成SQL查询语句，不要包含任何解释、注释或其他文本。直接输出SQL语句。"
                    sql = generate_single_sql(enhanced_prompt, model, tokenizer, device)
                    
                    sql = extract_sql_from_text(sql)
                    
                    results.append({
                        'sql_id': prompt['sql_id'],
                        'sql': sql
                    })
                except Exception as e2:
                    print(f"生成SQL时出错 (SQL ID: {prompt['sql_id']}): {e2}")
                    results.append({
                        'sql_id': prompt['sql_id'],
                        'sql': f"ERROR: {str(e2)}"
                    })
    
    # 保存结果
    if output_file is None:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"generated_sql_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"成功生成 {len(results)} 个SQL语句，已保存到: {output_file}")
    return len(results), results

def generate_single_sql(prompt_text, model, tokenizer, device=None):
    """
    生成单个SQL语句
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 确保模型在正确的设备上
    model = model.to(device)
    model.eval()
    
    # 设置生成参数
    generation_config = GenerationConfig(
        max_new_tokens=512,
        do_sample=False,
        temperature=0.1,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=1,
    )
    
    # 编码输入
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成SQL
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            return_dict_in_generate=True
        )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
    # 提取SQL部分
    sql = generated_text[len(prompt_text):].strip()
    
    return sql

# 加载模型和tokenizer
model_path = config.model_path
print(f"从本地加载模型: {model_path}")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"模型路径不存在: {model_path}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # 如果tokenizer没有pad_token，设置为eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 加载LoRA适配器
    model = PeftModel.from_pretrained(base_model, "final")
    print("模型加载成功！")
    
except Exception as e:
    print(f"模型加载失败: {e}")
    raise

# 配置路径
prompts_file = "sql_prompts.txt"
output_file = "generated_sql_results.json"

# 设置批量大小
if torch.cuda.is_available():
    batch_size = 2  # 使用较小的批量大小确保稳定性
else:
    batch_size = 1

print(f"设置批量大小为: {batch_size}")

# 清空GPU缓存
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 批量生成SQL
try:
    count, results = batch_generate_sql(
        prompts_file, 
        model, 
        tokenizer, 
        output_file, 
        batch_size=batch_size
    )
    
    if count > 0:
        print(f"SQL生成完成！共生成 {count} 个SQL语句。")
        for i, result in enumerate(results):
            print(f"\n--- 结果 {i+1} (SQL ID: {result['sql_id']}) ---")
            print(f"生成的SQL: {result['sql']}")
    else:
        print("SQL生成失败！")
        
except Exception as e:
    print(f"生成过程中发生错误: {e}")
    import traceback
    traceback.print_exc()
