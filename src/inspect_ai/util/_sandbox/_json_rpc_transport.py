"""JSON-RPC transport implementation for sandbox environments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from inspect_ai._util._json_rpc import (
    JSONRPCParamsType,
    JSONRPCTransport,
    create_json_rpc_request,
    rpc_call_description,
)

if TYPE_CHECKING:
    from .environment import SandboxEnvironment


SERVER_SOCKET = "/tmp/sandbox-tools/sandbox-tools.sock"

# Tools handled in-process by ``inspect-sandbox-tools exec`` (not by the
# server). All other methods are forwarded to the server's unix-socket HTTP
# endpoint, so we can skip the ~50MB PyInstaller decompress and call the
# socket directly via curl.
_IN_PROCESS_METHODS = frozenset({"text_editor", "version"})


class SandboxJSONRPCTransport(JSONRPCTransport):
    """A transport that uses a sandbox for RPC communication.

    This class implements the JSONRPCTransport protocol. The timeout and user
    parameters are passed via transport_extra_args in the __call__ method.
    """

    def __init__(
        self,
        sandbox: SandboxEnvironment,
        cli: str,
    ):
        """Initialize a new SandboxJSONRPCTransport.

        Args:
            sandbox: The sandbox environment to use.
            cli: The path to the cli available in the sandbox.
        """
        self.sandbox = sandbox
        self.cli = cli

    async def __call__(
        self,
        method: str,
        params: JSONRPCParamsType,
        is_notification: bool,
        **transport_extra_args: Any,
    ) -> str:
        """Execute an RPC request using the sandbox transport.

        Args:
            method: The JSON-RPC method to call.
            params: The parameters for the JSON-RPC method.
            is_notification: Whether this is a notification (no response expected).
            **transport_extra_args: Additional parameters including timeout and user.

        Returns:
            The response from the RPC call.

        Raises:
            RuntimeError: If the sandbox execution fails.
        """
        if method in _IN_PROCESS_METHODS:
            cmd = [self.cli, "exec"]
        else:
            cmd = ["curl", "-sSf", "--unix-socket", SERVER_SOCKET, "-d", "@-", "http://localhost/"]
        exec_result = await self.sandbox.exec(
            cmd,
            input=create_json_rpc_request(method, params, is_notification),
            timeout=transport_extra_args.get("timeout", None),
            timeout_retry=transport_extra_args.get("timeout_retry", True),
            user=transport_extra_args.get("user", None),
            concurrency=transport_extra_args.get("concurrency", True),
        )

        if not exec_result.success:
            raise RuntimeError(
                f"Sandbox.exec failure executing {rpc_call_description(method, params)}: {exec_result.stderr}"
            )
        return exec_result.stdout
