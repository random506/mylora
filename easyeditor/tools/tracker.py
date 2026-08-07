import os
from dotenv import load_dotenv, find_dotenv

try:
    import wandb
except ImportError:
    wandb = None

try:
    import swanlab
except ImportError:
    swanlab = None


class ExperimentTracker:
    """单例实验追踪器，通过类名直接调用

    """
    _instance = None

    # ── 类级状态 ──
    _mode: bool = True
    _use_wandb: bool = False
    _use_swanlab: bool = False
    _project: str = ""
    _name: str = ""

    @classmethod
    def init(cls,
             project: str,
             name: str = None,
             config: dict = None,
             tracker_type: str = None,
             mode: bool = True):
        """初始化单例追踪器。

        :param project: 项目名称
        :param name: 实验/Run名称
        :param config: 超参数配置字典
        :param tracker_type: "wandb", "swanlab", 或 "none"
        :param mode: 是否启用追踪
        """
        load_dotenv(find_dotenv())

        cls._mode = mode
        if tracker_type is None:
            cls._use_wandb = False
            cls._use_swanlab = False
        else:
            ttype = tracker_type.lower()
            cls._use_wandb = (ttype == "wandb")
            cls._use_swanlab = (ttype == "swanlab")

        cls._project = project
        cls._name = name or ""
        cls._config = config or {}

        if cls._mode:
            cls._setup_backend()
            cls._launch_run()

        cls._instance = cls.__new__(cls)

    @classmethod
    def _setup_backend(cls):
        """登录对应的后端服务。"""
        if cls._use_wandb:
            if wandb is None:
                raise ImportError("未安装 wandb 库!")
            key = os.getenv("WANDB_API_KEY")
            if not key:
                raise ValueError(".env 中未找到 WANDB_API_KEY")
            wandb.login(key=key)

        elif cls._use_swanlab:
            if swanlab is None:
                raise ImportError("未安装 swanlab 库!")
            key = os.getenv("SWANLAB_API_KEY")
            if not key:
                raise ValueError(".env 中未找到 SWANLAB_API_KEY")
            swanlab.login(api_key=key)

    @classmethod
    def _launch_run(cls):
        """启动对应的实验 run。"""
        if cls._use_wandb:
            wandb.init(project=cls._project, name=cls._name, config=cls._config)
            print(f"Wandb initialized -> Project: {cls._project}, Run: {cls._name}")
        elif cls._use_swanlab:
            swanlab.init(project=cls._project, experiment_name=cls._name, config=cls._config)
            print(f"SwanLab initialized -> Project: {cls._project}, Experiment: {cls._name}")

    @classmethod
    def log(cls, metrics: dict, step: int = None):
        """记录指标。"""
        if not cls._mode:
            return
        if cls._use_wandb:
            wandb.log(metrics, step=step)
        elif cls._use_swanlab:
            swanlab.log(metrics, step=step)

    @classmethod
    def finish(cls):
        """结束当前 run。"""
        if not cls._mode:
            return
        if cls._use_wandb:
            wandb.finish()
        elif cls._use_swanlab:
            if hasattr(swanlab, 'finish'):
                swanlab.finish()

