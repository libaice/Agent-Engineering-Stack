class DeterministicReActPlanner:
    def __init__(self, model: str):
        self.model = model
        self.system_prompt = ""


class ReActSupportAgent:
    def __init__(
        self,
        planner: DeterministicReActPlanner,
        max_steps: int = 8,
    ) -> None:
        pass


def main():
    agent = ReActSupportAgent(
        planner=DeterministicReActPlanner(),
        max_steps=8,
    )

    examples = [
        "我的订单 10086 已扣款，但是后台显示未支付，怎么办？",
        "你们的退款政策是什么？",
        "我的订单 99999 怎么查不到？",
    ]


if __name__ == "__main__":
    main()
