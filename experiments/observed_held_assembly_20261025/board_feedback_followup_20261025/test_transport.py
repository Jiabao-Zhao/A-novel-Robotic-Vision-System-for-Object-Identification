import json
import threading
import time

from .transport import CompleteResponseClient


def test_incomplete_response_waits_without_resending_command(tmp_path):
    client = CompleteResponseClient(tmp_path)
    errors = []

    def server():
        try:
            deadline = time.monotonic()+3
            while not list(client.queue.glob("*.request.json")):
                if time.monotonic() > deadline:
                    raise AssertionError("No request appeared")
                time.sleep(.005)
            request = next(client.queue.glob("*.request.json"))
            response = request.with_name(request.name.replace(".request.json", ".response.json"))
            response.write_text('{"ok":')
            time.sleep(.15)
            response.write_text(json.dumps({"ok": True, "robot_feedback": {"test": 1}}))
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=server)
    thread.start()
    result = client.call("move", phase="test_no_physics")
    thread.join(timeout=3)
    assert not thread.is_alive() and not errors
    assert result["robot_feedback"] == {"test": 1}
    assert len(list(client.queue.glob("*.request.json"))) == 1
    assert client.calls[0]["incomplete_response_reads_retried"] > 0
    assert not client.calls[0]["physical_command_resent"]
