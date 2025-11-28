import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import logging
from typing import List, Dict, Tuple
import time
import json
import config

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OptimizedQwenColumnGenerator:
    def __init__(self, model_path: str):
        """初始化Qwen模型"""
        logger.info(f"加载Qwen模型从: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        logger.info(f"模型加载完成")

    def extract_sql_id(self, prompt_content: str) -> str:
        """从prompt内容中提取SQL ID"""
        match = re.search(r'SQL ID:\s*([^\n]+)', prompt_content)
        if match:
            return match.group(1).strip()
        return f"unknown_{hash(prompt_content) % 10000}"

    def build_optimized_prompt(self, prompt_content: str) -> str:
        """构建优化的prompt，保持原始内容完整性"""
        system_message = """你是一个专业的SQL和数据仓库专家。请严格按照以下格式返回结果：

列名是：[表名.列名, 表名.列名, 表名.列名]
要求：
1. 只返回上述两个列表，不要有任何其他文本
2. 表名和列名必须与提供的schema完全一致
3. 确保使用完整的表名和列名
4. 严格按照格式输出，不要添加任何解释"""

        formatted_prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{prompt_content}<|im_end|>\n<|im_start|>assistant\n列名是：["

        return formatted_prompt

    def parse_optimized_response(self, response: str) -> Tuple[List[str], List[str]]:
        """解析优化的响应，包含列名和外键"""
        columns = []
        foreign_keys = []

        # 查找列名列表
        columns_pattern = r'列名是：\s*\[([^\]]+)\]'
        columns_match = re.search(columns_pattern, response)

        if columns_match:
            columns_content = columns_match.group(1).strip()
            # 使用更灵活的分割方式
            items = re.split(r',\s*(?=[^,]*(?:\.[^,]*)?$)', columns_content)
            for item in items:
                item = item.strip()
                if '.' in item and not item.endswith('.'):
                    columns.append(item)

        # 查找外键列表
        fk_pattern = r'外键是：\s*\[([^\]]+)\]'
        fk_match = re.search(fk_pattern, response)

        if fk_match:
            fk_content = fk_match.group(1).strip()
            # 使用更灵活的分割方式
            items = re.split(r',\s*(?=[^,]*(?:=[^,]*)?$)', fk_content)
            for item in items:
                item = item.strip()
                if '=' in item:
                    foreign_keys.append(item)

        logger.info(f"解析到 {len(columns)} 个列名和 {len(foreign_keys)} 个外键")
        return columns, foreign_keys

    def generate_with_optimized_params(self, prompt_content: str, max_retries: int = 3) -> Tuple[List[str], List[str]]:
        """使用优化的参数生成列名和外键"""
        for attempt in range(max_retries):
            try:
                formatted_prompt = self.build_optimized_prompt(prompt_content)

                # 检查prompt长度
                prompt_tokens = len(self.tokenizer.encode(formatted_prompt))
                logger.info(f"Prompt token数量: {prompt_tokens}")

                # 动态调整参数
                generation_params = self.get_optimized_generation_params(attempt, prompt_tokens)

                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    max_length=min(4096, prompt_tokens + 500),  # 动态调整最大长度
                    truncation=True,
                    padding=True
                ).to(self.device)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        **generation_params
                    )

                full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # 提取assistant的回复部分
                response_start = formatted_prompt.find("列名是：[")
                response = full_response[response_start:] if response_start != -1 else full_response

                logger.debug(f"第 {attempt + 1} 次尝试完整响应: {response}")

                # 解析响应
                columns, foreign_keys = self.parse_optimized_response(response)

                if columns or foreign_keys:
                    logger.info(f"第 {attempt + 1} 次尝试成功")
                    return columns, foreign_keys
                else:
                    logger.warning(f"第 {attempt + 1} 次尝试未解析到内容")
                    # 记录部分响应用于调试
                    if len(response) > 100:
                        logger.debug(f"响应片段: {response[:100]}...")

            except torch.cuda.OutOfMemoryError:
                logger.warning(f"第 {attempt + 1} 次尝试GPU内存不足，清理缓存")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # 减少生成长度
                if attempt < max_retries - 1:
                    continue
            except Exception as e:
                logger.error(f"第 {attempt + 1} 次尝试出错: {e}")

        return [], []

    def get_optimized_generation_params(self, attempt: int, prompt_tokens: int) -> Dict:
        """根据尝试次数和prompt长度优化生成参数"""
        # 基础参数
        base_params = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # 根据prompt长度调整生成长度
        max_new_tokens = 4096

        if attempt == 0:
            # 第一次尝试：平衡确定性和创造性
            base_params.update({
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
            })
        elif attempt == 1:
            # 第二次尝试：增加确定性
            base_params.update({
                "max_new_tokens": max_new_tokens ,
                "do_sample": False,
                "temperature": 0.1,
                "repetition_penalty": 1.05,
            })
        else:
            # 第三次尝试：增加创造性
            base_params.update({
                "max_new_tokens": max_new_tokens ,
                "do_sample": True,
                "temperature": 0.5,
                "top_p": 0.95,
                "repetition_penalty": 1.2,
                "num_beams": 3,
                "early_stopping": True,
            })

        logger.debug(f"第 {attempt + 1} 次尝试生成参数: {base_params}")
        return base_params

    def parse_prompts_with_sql_id(self, file_path: str) -> List[Dict]:
        """解析prompt文件并保留SQL ID"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        prompt_blocks = re.split(r'=== Prompt \d+ ===', content)
        prompts = []

        for i, block in enumerate(prompt_blocks[1:], 1):
            block = block.strip()
            if not block:
                continue

            sql_id = self.extract_sql_id(block)

            prompts.append({
                'id': i,
                'sql_id': sql_id,
                'content': block
            })

        logger.info(f"解析到 {len(prompts)} 个prompt")
        return prompts

    def process_with_optimized_params(self, prompt_file: str, output_file: str = None):
        """使用优化参数处理所有prompt"""
        prompts = self.parse_prompts_with_sql_id(prompt_file)
        results = {}

        logger.info(f"开始处理 {len(prompts)} 个prompt，使用优化参数")
        start_time = time.time()

        success_count = 0
        empty_count = 0

        for i, prompt in enumerate(prompts, 1):
            logger.info(f"处理进度: {i}/{len(prompts)} - SQL ID: {prompt['sql_id']}")

            # 生成列名和外键
            columns, foreign_keys = self.generate_with_optimized_params(prompt['content'])

            # 记录结果
            results[prompt['sql_id']] = {
                'prompt_id': prompt['id'],
                'columns': columns,
                'foreign_keys': foreign_keys,
                'timestamp': time.time(),
                'column_count': len(columns),
                'fk_count': len(foreign_keys),
                'success': bool(columns or foreign_keys)
            }

            if columns or foreign_keys:
                success_count += 1
                status = "成功"
                if columns and foreign_keys:
                    status += " (列名+外键)"
                elif columns:
                    status += " (仅列名)"
                else:
                    status += " (仅外键)"

                logger.info(f"? SQL ID {prompt['sql_id']}: {status} - {len(columns)}列名, {len(foreign_keys)}外键")

                if columns and len(columns) <= 5:
                    logger.info(f"  列名: {columns}")
                elif columns:
                    logger.info(f"  前3个列名: {columns[:3]}...")

                if foreign_keys:
                    logger.info(f"  外键: {foreign_keys}")
            else:
                empty_count += 1
                logger.error(f"? SQL ID {prompt['sql_id']}: 未生成任何内容")

            # 动态调整延迟
            delay = 0.5 if torch.cuda.is_available() else 0.1
            time.sleep(delay)

        end_time = time.time()

        # 详细统计
        total_columns = sum(len(result['columns']) for result in results.values())
        total_fks = sum(len(result['foreign_keys']) for result in results.values())
        total_success = sum(1 for result in results.values() if result['success'])

        logger.info(f"\n{'=' * 60}")
        logger.info("优化参数处理统计:")
        logger.info(f"总prompt数量: {len(prompts)}")
        logger.info(f"成功生成内容: {total_success}")
        logger.info(f"未生成内容: {empty_count}")
        logger.info(f"总列名数量: {total_columns}")
        logger.info(f"总外键数量: {total_fks}")
        logger.info(f"成功率: {total_success / len(prompts) * 100:.1f}%")
        logger.info(f"平均每个成功prompt的列名: {total_columns / max(1, total_success):.1f}")
        logger.info(f"平均每个成功prompt的外键: {total_fks / max(1, total_success):.1f}")
        logger.info(f"总耗时: {end_time - start_time:.2f}秒")
        logger.info(f"平均每个prompt耗时: {(end_time - start_time) / len(prompts):.2f}秒")

        # 保存结果
        if output_file:
            self.save_optimized_results(results, output_file)

        return results

    def save_optimized_results(self, results: Dict, output_file: str):
        """保存优化结果"""
        # JSON格式
        json_output = output_file.replace('.txt', '.json')
        with open(json_output, 'w', encoding='utf-8') as f:
            simple_results = {}
            for sql_id, result in results.items():
                simple_results[sql_id] = {
                    'prompt_id': result['prompt_id'],
                    'columns': result['columns'],
                    'foreign_keys': result['foreign_keys'],
                    'column_count': result['column_count'],
                    'fk_count': result['fk_count'],
                    'success': result['success']
                }
            json.dump(simple_results, f, ensure_ascii=False, indent=2)

        # 文本格式
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("优化参数列名和外键生成结果\n")
            f.write("=" * 60 + "\n\n")

            successful = sum(1 for r in results.values() if r['success'])
            total_columns = sum(len(r['columns']) for r in results.values())
            total_fks = sum(len(r['foreign_keys']) for r in results.values())

            f.write(f"统计信息:\n")
            f.write(f"- 总SQL ID数量: {len(results)}\n")
            f.write(f"- 成功生成的SQL ID: {successful}\n")
            f.write(f"- 生成的列名总数: {total_columns}\n")
            f.write(f"- 生成的外键总数: {total_fks}\n")
            f.write(f"- 成功率: {successful / len(results) * 100:.1f}%\n\n")

            f.write("详细结果:\n")
            f.write("-" * 60 + "\n\n")

            for sql_id in sorted(results.keys()):
                result = results[sql_id]
                status = "成功" if result['success'] else "失败"
                f.write(f"SQL ID: {sql_id} (Prompt ID: {result['prompt_id']}) - {status}\n")
                f.write(f"列名数量: {result['column_count']}\n")
                f.write(f"外键数量: {result['fk_count']}\n")

                if result['columns']:
                    f.write("列名列表:\n")
                    for column in result['columns']:
                        f.write(f"  {column}\n")
                else:
                    f.write("未生成任何列名\n")

                if result['foreign_keys']:
                    f.write("外键列表:\n")
                    for fk in result['foreign_keys']:
                        f.write(f"  {fk}\n")
                else:
                    f.write("未生成任何外键\n")

                f.write("\n" + "-" * 40 + "\n\n")

        logger.info(f"结果已保存到: {output_file} 和 {json_output}")


def analyze_failed_prompts(generator: OptimizedQwenColumnGenerator, results: Dict, prompts: List[Dict]):
    """分析失败的prompt"""
    failed_sql_ids = [sql_id for sql_id, result in results.items() if not result['success']]

    if not failed_sql_ids:
        logger.info("没有失败的prompt")
        return

    logger.info(f"分析 {len(failed_sql_ids)} 个失败的prompt")

    for sql_id in failed_sql_ids[:3]:  # 只分析前3个
        prompt = next(p for p in prompts if p['sql_id'] == sql_id)
        logger.info(f"分析失败SQL ID: {sql_id}")

        # 检查prompt长度
        prompt_length = len(prompt['content'])
        token_count = len(generator.tokenizer.encode(prompt['content']))
        logger.info(f"Prompt长度: {prompt_length} 字符, {token_count} tokens")

        # 检查问题类型
        problem_match = re.search(r'问题:\s*(.+?)(?=\n\n)', prompt['content'])
        if problem_match:
            problem = problem_match.group(1)
            logger.info(f"问题内容: {problem[:100]}...")

        # 检查表数量
        table_count = len(re.findall(r'表\s+[a-zA-Z_][a-zA-Z0-9_]*\s+的完整结构:', prompt['content']))
        logger.info(f"涉及表数量: {table_count}")


MODEL_PATH = config.model_path
PROMPT_FILE = "all_schema_linking_prompts.txt"
OUTPUT_FILE = "optimized_columns_fks.txt"

try:
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

    generator = OptimizedQwenColumnGenerator(MODEL_PATH)

    # 处理所有prompt
    results = generator.process_with_optimized_params(PROMPT_FILE, OUTPUT_FILE)

    # 分析失败案例
    prompts = generator.parse_prompts_with_sql_id(PROMPT_FILE)
    analyze_failed_prompts(generator, results, prompts)

    # 最终统计
    final_success = sum(1 for result in results.values() if result['success'])
    logger.info(f"最终处理完成，成功率: {final_success / len(results) * 100:.1f}%")

    # 输出失败的SQL ID
    failures = [sql_id for sql_id, result in results.items() if not result['success']]
    if failures:
        logger.warning(f"以下SQL ID未生成任何内容: {failures}")

except Exception as e:
    logger.error(f"程序执行出错: {e}", exc_info=True)
