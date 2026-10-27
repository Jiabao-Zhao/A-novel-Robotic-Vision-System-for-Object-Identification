"""Read a completed file response without resending a physical command."""
import json
import time
import uuid

from ..plant import PlantClient, write


class CompleteResponseClient(PlantClient):
    def call(self, action, **arguments):
        token = uuid.uuid4().hex
        request = self.queue/(token+".request.json")
        response = self.queue/(token+".response.json")
        temporary = request.with_suffix(".tmp")
        write(temporary, {"action": action, **arguments})
        temporary.rename(request)
        started = time.perf_counter()
        incomplete_reads = 0
        while True:
            if time.perf_counter()-started > 600:
                raise TimeoutError("Image/robot execution server did not provide a complete response")
            if response.exists():
                try:
                    value = json.loads(response.read_text())
                    break
                except json.JSONDecodeError:
                    incomplete_reads += 1
                    if incomplete_reads > 100:
                        raise RuntimeError("Execution response remained incomplete for five seconds")
            time.sleep(.05)
        self.calls.append({"action": action, "request": str(request.relative_to(self.run)),
            "response": str(response.relative_to(self.run)),
            "wall_seconds": time.perf_counter()-started,
            "incomplete_response_reads_retried": incomplete_reads,
            "physical_command_resent": False})
        write(self.run/"client_rpc_log.json", self.calls)
        if not value.get("ok"):
            raise RuntimeError(value.get("error", "Execution adapter failed"))
        return value
