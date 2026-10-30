const { createInterface } = require("node:readline");
const frame = value => process.stdout.write(JSON.stringify(value) + "\n");

async function main() {
  for await (const line of createInterface({ input: process.stdin })) {
    const message = JSON.parse(line);
    if (message.type === "control_request" && message.request.subtype === "initialize") {
      frame({ type: "control_response", response: {
        subtype: "success", request_id: message.request_id, response: {},
      } });
    } else if (message.type === "user") {
      const chunk = Buffer.alloc(64 * 1024, "W");
      chunk.write("Private offline stderr diagnostic\n");
      // Wait for each write so the result proves the pipe drained, even on async stdio.
      for (let i = 0; i < 256; i++) {
        await new Promise((resolve, reject) => {
          process.stderr.write(chunk, error => error ? reject(error) : resolve());
        });
      }
      frame({ type: "result", subtype: "success", result: "stderr drained", stderr_bytes: chunk.length * 256,
        is_error: false, num_turns: 1, session_id: "offline" });
    }
  }
}

main().catch(() => { process.exitCode = 1; });
