import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import asyncio
import config


class UltraFastLoRAQwenSQLGenerator:
    def __init__(self, base_model_path: str, lora_model_path: str, max_batch_size: int = 16):
        """
        超快速LoRA Qwen SQL生成器

        Args:
            base_model_path: 基础模型路径
            lora_model_path: LoRA适配器路径
            max_batch_size: 最大批处理大小
        """
        self.max_batch_size = max_batch_size

        logger.info("超快速加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
            padding_side="left"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 使用量化加速（如果支持）
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",
            low_cpu_mem_usage=True
        )

        self.model = PeftModel.from_pretrained(base_model, lora_model_path)
        self.model.eval()

        logger.info("超快速LoRA Qwen模型加载完成")

    def build_prompt_fast(self, question: str, columns: List[str], table_list: List[str], knowledge: str = "") -> str:
        """极简提示词"""
        return f"SQL for: {question} | Tables: {table_list} | Columns: {columns} | SQL:"

    def ultra_fast_generate(self, prompts: List[str]) -> List[str]:
        """超快速生成"""
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=384
        )

        inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=128,  # 减少生成长度
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_beams=1,  # 使用单束搜索加速
                    early_stopping=True
                )

        # 快速解码
        sqls = []
        for i, output in enumerate(outputs):
            input_len = inputs['input_ids'][i].shape[0]
            sql = self.tokenizer.decode(output[input_len:], skip_special_tokens=True)
            sqls.append(sql.strip())

        return sqls

    def process_ultra_fast(self, columns_file: str, questions_file: str, output_file: str):
        """超快速处理"""
        with open(columns_file, 'r', encoding='utf-8') as f:
            columns_data = json.load(f)

        with open(questions_file, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)

        # 准备所有数据
        all_data = []
        for q in questions_data:
            sql_id = q["sql_id"]
            columns = columns_data.get(sql_id, {}).get("columns", [])
            prompt = self.build_prompt_fast(
                q["question"],
                columns,
                q.get("table_list", []),
                q.get("knowledge", "")
            )
            all_data.append((q, prompt))

        # 批量处理
        results = []
        for i in tqdm(range(0, len(all_data), self.max_batch_size), desc="超快速生成"):
            batch_data = all_data[i:i + self.max_batch_size]
            batch_items, batch_prompts = zip(*batch_data)

            batch_sqls = self.ultra_fast_generate(batch_prompts)

            for item, sql in zip(batch_items, batch_sqls):
                result = item.copy()
                result["generated_sql"] = sql
                results.append(result)

        # 保存
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"超快速处理完成！共处理 {len(results)} 个问题")


generator = UltraFastLoRAQwenSQLGenerator(
    base_model_path=config.model_path,
    lora_model_path="final",
    max_batch_size=16  # 根据GPU调整
)

generator.process_ultra_fast(
    "optimized_columns_fks.json",
    "final_dataset.json",
    "output_ultra_fast.json"
)