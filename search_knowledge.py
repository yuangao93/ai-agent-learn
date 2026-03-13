"""
向量检索测试脚本
输入一个问题，从知识库中找到最相关的段落
"""
import chromadb

def search(question: str, top_k: int=2):
    """
    从向量数据库中检索最相关的文本段
    :param question: 用户的问题
    :param top_k: 返回最相关的几段
    :return:
    """
    search_client = chromadb.PersistentClient(path="./chroma_db")
    search_collection = search_client.get_collection(name="knowledge")

    # 核心就这一行：ChromaDB自动把question转成向量，然后找最接近的段落
    results = search_collection.query(
        query_texts=[question],
        n_results=top_k
    )

    return results

if __name__ == "__main__":
    # 测试几个不同的问题，观察检索结果的差异
    test_questions = [
        "Agent有哪些关键能力？",
        "什么是多Agent系统？",
        "AI Agent有哪些应用场景？",
    ]

    for q in test_questions:
        print(f"\n{'='*50}")
        print(f"问题：{q}")
        print(f"{'='*50}")

        results = search(q)

        for i, (doc, distance) in enumerate(zip(results["documents"][0], results["distances"][0])):
            print(f"\n[第{i+1}相关] 距离：{distance:.4f}")
            print(f"内容：{doc[:150]}...")