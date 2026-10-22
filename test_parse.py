from rag_utils import load_uploaded_file_text

file_path = "使用说明.pdf"  # 换成你项目里真实存在的文件路径

with open(file_path, "rb") as f:
    file_bytes = f.read()

text = load_uploaded_file_text(file_path, file_bytes)

print("文本长度：", len(text))
print("文本预览：")
print(text[:300])