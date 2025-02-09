# LLM-JP Fine-tuning Project

このプロジェクトは、日本語 LLM (`llm-jp/llm-jp-3-13b`) の Instruction-Tuning を行うためのスクリプト群です。

## 📂 プロジェクト構成

## 🚀 セットアップ方法
```bash
git clone https://github.com/your_username/llm-jp-finetuning.git
cd llm-jp-finetuning
pip install -r requirements.txt

学習の実行
python train.py

モデルの評価
python evaluate.py

Hugging Face へアップロード
python save_model.py
