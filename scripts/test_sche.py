import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from typing import List, Dict, Any
import logging
from tqdm import tqdm
import config
import os

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRAQwenSQLGenerator:
    def __init__(self, base_model_path: str, lora_model_path: str, device: str = "cuda"):
        """
        初始化LoRA Qwen SQL生成器

        Args:
            base_model_path: 基础模型路径
            lora_model_path: LoRA适配器路径
            device: 设备类型
        """
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        logger.info(f"使用设备: {self.device}")

        # 加载tokenizer和基础模型
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto"
        )

        # 加载LoRA适配器
        self.model = PeftModel.from_pretrained(base_model, lora_model_path)
        self.model.to(self.device)
        self.model.eval()

        logger.info("LoRA Qwen模型加载完成")

    def build_prompt(self, question: str, columns: List[str], table_list: List[str], knowledge: str = "") -> str:
        """
        构建提示词

        Args:
            question: 问题文本
            columns: 列列表
            table_list: 表列表
            knowledge: 相关知识

        Returns:
            构建好的提示词
        """
        # 格式化列信息
        columns_str = "\n".join([f"- {col}" for col in columns])
        tables_str = "\n".join([f"- {table}" for table in table_list])

        prompt = f"""你是一个SQL专家。请根据以下信息生成SQL查询语句。

数据库表:
{tables_str}

相关字段:
{columns_str}

业务知识:
{knowledge}

问题:
{question}

请生成标准的SQL查询语句，只输出SQL代码，不要其他解释:"""

        return prompt

    def generate_sql(self, prompt: str, max_length: int = 512, temperature: float = 0.1) -> str:
        """
        生成SQL查询

        Args:
            prompt: 输入提示词
            max_length: 最大生成长度
            temperature: 生成温度

        Returns:
            生成的SQL语句
        """
        try:
            # 编码输入
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 生成SQL
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )

            # 解码输出
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 提取SQL部分（去除提示词）
            sql = generated_text[len(prompt):].strip()

            return sql

        except Exception as e:
            logger.error(f"生成SQL时出错: {e}")
            return ""

    def process_json_files(self, columns_file: str, questions_file: str, output_file: str):
        """
        处理JSON文件并生成SQL

        Args:
            columns_file: 包含列信息的JSON文件路径
            questions_file: 包含问题的JSON文件路径
            output_file: 输出文件路径
        """
        try:
            # 读取列信息文件
            with open(columns_file, 'r', encoding='utf-8') as f:
                columns_data = json.load(f)

            # 读取问题文件
            with open(questions_file, 'r', encoding='utf-8') as f:
                questions_data = json.load(f)

            logger.info(f"加载了 {len(columns_data)} 个列配置和 {len(questions_data)} 个问题")

            # 处理每个问题
            results = []
            for question_item in tqdm(questions_data, desc="生成SQL"):
                sql_id = question_item.get("sql_id")

                # 在列数据中查找对应的列信息
                column_info = columns_data.get(sql_id, {})
                columns = column_info.get("columns", [])

                # 构建提示词
                prompt = self.build_prompt(
                    question=question_item.get("question", ""),
                    columns=columns,
                    table_list=question_item.get("table_list", []),
                    knowledge=question_item.get("knowledge", "")
                )

                # 生成SQL
                logger.info(f"为 {sql_id} 生成SQL...")
                generated_sql = self.generate_sql(prompt)

                # 添加到结果中
                result_item = question_item.copy()
                result_item["generated_sql"] = generated_sql
                result_item["columns_used"] = columns
                results.append(result_item)

                logger.info(f"{sql_id} 生成的SQL: {generated_sql[:100]}...")

            # 保存结果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            logger.info(f"结果已保存到: {output_file}")

        except Exception as e:
            logger.error(f"处理文件时出错: {e}")
            raise


BASE_MODEL_PATH = config.model_path # 根据实际情况修改基础模型路径
LORA_MODEL_PATH = "final"  # 修改为您的LoRA适配器路径
COLUMNS_FILE = "optimized_columns_fks.json"  # 列信息文件
QUESTIONS_FILE = "final_dataset.json"  # 问题文件
OUTPUT_FILE = "output_with_sql.json"  # 输出文件

# 初始化生成器
generator = LoRAQwenSQLGenerator(
    base_model_path=BASE_MODEL_PATH,
    lora_model_path=LORA_MODEL_PATH,
    device="cuda"  # 使用GPU加速
)

# 处理文件
generator.process_json_files(
    columns_file=COLUMNS_FILE,
    questions_file=QUESTIONS_FILE,
    output_file=OUTPUT_FILE
)
