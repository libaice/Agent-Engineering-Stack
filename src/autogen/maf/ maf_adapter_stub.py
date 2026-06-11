class AgentRuntimeAdapter:
    def run(self, question: str) -> dict:
        raise NotImplementedError


class MiniAutoGenAdapter(AgentRuntimeAdapter):
    def run(self, question: str) -> dict:
        pass


class MicrosoftAgentFrameworkAdapter(AgentRuntimeAdapter):
    """
    TODO:
    Implement using Microsoft Agent Framework:
    - Agents
    - Sessions
    - Tools
    - Workflows
    - Telemetry
    """
