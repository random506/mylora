import os
from dotenv import load_dotenv ,find_dotenv
try:
    import wandb
except ImportError:
    wandb = None

try:
    import swanlab
except ImportError:
    swanlab = None


class ExperimentTracker:
    def __init__(self, 
                 project: str, 
                 name: str = None, 
                 config: dict = None, 
                 tracker_type: str = None,
                 mode: bool = True):
        """
        :param project: 项目名称
        :param name: 实验/Run名称
        :param config: 超参数配置字典
        :param tracker_type: 决定使用哪个追踪器，可选值: "wandb", "swanlab", "none"
        """
        load_dotenv(find_dotenv())
        
        self.project = project
        self.name = name
        self.config = config or {}

        self.tracker_type = tracker_type.lower()
        
        self.use_wandb = (self.tracker_type == "wandb")
        self.use_swanlab = (self.tracker_type == "swanlab")
        self.mode = mode
        if self.mode:
            self._setup_trackers()

    def _setup_trackers(self):
        if self.use_wandb:
            if wandb is None:
                raise ValueError("未安装wandb库!")
            else:
                wandb_key = os.getenv("WANDB_API_KEY")
                if wandb_key:
                    wandb.login(key=wandb_key)
                else:
                    raise ValueError(".env 中未找到 WANDB_API_KEY")

        elif self.use_swanlab:
            if swanlab is None:
                raise ValueError("未安装swanlab库!")
            else:
                swanlab_key = os.getenv("SWANLAB_API_KEY")
                if swanlab_key:
                    # SwanLab 的 API Key 登录机制
                    swanlab.login(api_key=swanlab_key)
                else:
                    raise ValueError(".env 中未找到 SWANLAB_API_KEY")

    def init(self):
        if not self.mode:
            return
        if self.use_wandb:
            wandb.init(
                project=self.project,
                name=self.name,
                config=self.config
            )
            print(f"Wandb initialized -> Project: {self.project}, Run: {self.name}")

        elif self.use_swanlab:
            swanlab.init(
                project=self.project,
                experiment_name=self.name, 
                config=self.config
            )
            print(f"SwanLab initialized -> Project: {self.project}, Experiment: {self.name}")

    def log(self, metrics: dict, step: int = None):
        if not self.mode:
            return
        if self.use_wandb:
            wandb.log(metrics, step=step)
            
        elif self.use_swanlab:
            swanlab.log(metrics, step=step)

    def finish(self):
        if not self.mode:
            return
        if self.use_wandb:
            wandb.finish()
            
        elif self.use_swanlab:
            if hasattr(swanlab, 'finish'):
                swanlab.finish()


def test():
    demo = ExperimentTracker("111","111",{},"wandb")
    demo.init()
    for i in range(100):
        demo.log({"demo":i+1})
    demo.finish()


if __name__ == "__main__":
    test()
