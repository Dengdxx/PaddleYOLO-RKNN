# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Modifications Copyright (C) 2026 Dengdxx <dengdx@tju.edu.cn>

"""
@file
@brief Git 版本信息工具：读取当前仓库 Tag/Hash 供日志记录。
@details
提供 `git_describe()` 函数，在训练日志中记录代码版本，
便于结果复现时追溯对应代码状态。
"""

import paddle

from functools import cached_property
from pathlib import Path


class GitRepo:
    """表示 local Git repository，并暴露 branch、commit 与 remote metadata。

    该类从给定 path 向上搜索 .git entry 来发现 repository root，解析实际 .git directory（包括 worktrees），
    并直接从磁盘文件读取 Git metadata。它不调用 git binary，因此可用于受限环境。所有 metadata properties
    都会 lazy resolve 并缓存；如需刷新 state，请构造新 instance。

    属性:
        root (Path | None): 包含 .git entry 的 repository root directory；不在 repository 中时为 None。
        gitdir (Path | None): resolved .git directory path；处理 worktrees；无法解析时为 None。
        head (str | None): HEAD 的原始内容；detached HEAD 时为 SHA，branch heads 时为 "ref: <refname>"。
        is_repo (bool): 给定 path 是否位于 Git repository 内。
        branch (str | None): HEAD 指向 branch 时的 current branch name；detached HEAD 或 non-repo 时为 None。
        commit (str | None): HEAD 的 current commit SHA；无法确定时为 None。
        origin (str | None): 从 gitdir/config 读取的 "origin" remote URL；未设置或不可用时为 None。

    示例:
        从 current working directory 初始化并读取 metadata
        >>> from pathlib import Path
        >>> repo = GitRepo(Path.cwd())
        >>> repo.is_repo
        True
        >>> repo.branch, repo.commit[:7], repo.origin
        ('main', '1a2b3c4', 'https://example.com/owner/repo.git')

    说明:
        - 通过读取 HEAD、packed-refs 与 config files 解析 metadata；不使用 subprocess calls。
        - 使用 cached_property 在首次访问时缓存 properties；如需反映 repository changes，请重新创建 object。
    """

    def __init__(self, path: Path = Path(__file__).resolve()):
        """从 starting path 发现 repository root，初始化 Git repository context。

        参数:
            path (Path, optional): 用作查找 repository root 起点的 file 或 directory path。
        """
        self.root = self._find_root(path)
        self.gitdir = self._gitdir(self.root) if self.root else None

    @staticmethod
    def _find_root(p: Path) -> Path | None:
        """返回 repo root 或 None。"""
        return next((d for d in [p, *list(p.parents)] if (d / ".git").exists()), None)

    @staticmethod
    def _gitdir(root: Path) -> Path | None:
        """解析实际 .git directory（处理 worktrees）。"""
        g = root / ".git"
        if g.is_dir():
            return g
        if g.is_file():
            t = g.read_text(errors="ignore").strip()
            if t.startswith("gitdir:"):
                return (root / t.split(":", 1)[1].strip()).resolve()
        return None

    @staticmethod
    def _read(p: (Path | None)) -> str | None:
        """如果 file 存在则读取并 strip。"""
        return p.read_text(errors="ignore").strip() if p and p.exists() else None

    @cached_property
    def head(self) -> str | None:
        """HEAD file contents。"""
        return self._read(self.gitdir / "HEAD" if self.gitdir else None)

    def _ref_commit(self, ref: str) -> str | None:
        """ref 对应的 commit（处理 packed-refs）。"""
        rf = self.gitdir / ref
        if s := self._read(rf):
            return s
        pf = self.gitdir / "packed-refs"
        b = pf.read_bytes().splitlines() if pf.exists() else []
        tgt = ref.encode()
        for line in b:
            if line[:1] in (b"#", b"^") or b" " not in line:
                continue
            sha, name = line.split(b" ", 1)
            if name.strip() == tgt:
                return sha.decode()
        return None

    @property
    def is_repo(self) -> bool:
        """位于 git repo 内时为 True。"""
        return self.gitdir is not None

    @cached_property
    def branch(self) -> str | None:
        """current branch，或 None。"""
        if not self.is_repo or not self.head or not self.head.startswith("ref: "):
            return None
        ref = self.head[5:].strip()
        return ref[len("refs/heads/") :] if ref.startswith("refs/heads/") else ref

    @cached_property
    def commit(self) -> str | None:
        """current commit SHA，或 None。"""
        if not self.is_repo or not self.head:
            return None
        return self._ref_commit(self.head[5:].strip()) if self.head.startswith("ref: ") else self.head

    @cached_property
    def origin(self) -> str | None:
        """origin URL，或 None。"""
        if not self.is_repo:
            return None
        cfg = self.gitdir / "config"
        remote, url = None, None
        for s in (self._read(cfg) or "").splitlines():
            t = s.strip()
            if t.startswith("[") and t.endswith("]"):
                remote = t.lower()
            elif t.lower().startswith("url =") and remote == '[remote "origin"]':
                url = t.split("=", 1)[1].strip()
                break
        return url


if __name__ == "__main__":
    import time

    g = GitRepo()
    if g.is_repo:
        t0 = time.perf_counter()
        print(f"repo={g.root}\nbranch={g.branch}\ncommit={g.commit}\norigin={g.origin}")
        dt = (time.perf_counter() - t0) * 1000
        print(f"\n⏱️ Profiling: total {dt:.3f} ms")
