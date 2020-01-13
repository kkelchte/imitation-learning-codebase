import asyncio
import logging

from src.core.logger import cprint


async def run(cmd, logger: logging.Logger = None) -> (str, str):
    process = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()

    cprint(f'[{cmd!r} exited with {process.returncode}]', logger)
    if stdout:
        cprint(f'[stdout]\n{stdout.decode()}', logger)
    if stderr:
        cprint(f'[stderr]\n{stderr.decode()}', logger)
    return stdout, stderr
