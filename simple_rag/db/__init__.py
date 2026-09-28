"""`simple_rag.db` —— 数据库连接的生命周期。

对外只有 `Database` 一个名字。裸连接工厂 `open_connection` **不从这里导出** ——
两层入口会让人不知道该用哪个；要用请 `from simple_rag.db.database import open_connection`。
"""

from .database import Database

__all__ = ["Database"]
