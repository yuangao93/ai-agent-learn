"""
知识库构建脚本
把文档切成小段，转成向量，存进ChromaDB
"""
import chromadb
import os

from agent_v1 import client


# ============ 第一步：读取文档 ============
def load_documents(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

# ============ 第二步：切分文档 ============
def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list:
    """
    把长文本切成小段
    :param text: 长文本
    :param chunk_size: 每小段多少字
    :param overlap: 相邻段之间重叠多少字（避免语义被切断）
    :return: 所有段落列表
    """
    chunks = []
    start = 0 # 从0开始切
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip(): # 不是空段落才会加到chunks中
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ============ 第三步：存入向量数据库 ============
def build_vector_db(chunks: list, collection_name: str = "knowledge"):
    """
    把文本段存入ChromaDB
    ChromaDB会自动把文本转成向量（它内置了一个小型的embedding模型）
    :param chunks:
    :param collection_name:
    :return:
    """
    # 创建/连接本地数据库（数据存在./chroma_db文件夹中）
    chromadb_cilent = chromadb.PersistentClient(path="./chroma_db")

    # 如果已存在同名集合，先删除（重新构建）
    try:
        chromadb_cilent.delete_collection(name=collection_name)
    except:
        pass

    # 创建集合
    collection = chromadb_cilent.create_collection(name=collection_name)

    # 批量添加文档
    collection.add(
        documents=chunks,
        ids = [f"chunk_{i}" for i in range(len(chunks))]
    )
    print(f"知识库构建完成！共 {len(chunks)} 个文本段")
    return collection

# ============ 运行 ============
if __name__=="__main__":
    # 读取知识库文件
    text = load_documents("./knowledge.txt")
    print(f"文档总长度： {len(text)} 字")

    # 切分
    chunks = split_text(text)
    print(f"切分为 {len(chunks)} 段")

    # 查看前3段，确认切分效果
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- 第{i+1}段 ---")
        print(chunk[:100] + "...")

    # 存入向量数据库
    build_vector_db(chunks)
