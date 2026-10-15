# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:26:45 2026

@author: samth
"""

_debugpy_listening = False


def debugInPlace():
    """
    Pause in an attached VS Code debugger when running inside a Jupyter kernel.

    Outside an IPython kernel this does nothing. ``debugpy.listen`` may only be
    called once per process, so the listener is started on the first call
    and reused afterwards; every call still waits for a client and breaks.
    """
    global _debugpy_listening
    import os
    import sys

    is_jupyter = "ipykernel" in sys.modules
    if is_jupyter:
        try:
            import debugpy
            if not _debugpy_listening:
                # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
                try:
                    debugpy.listen(5678)
                except RuntimeError:
                    pass  # listen() was already called in this process (e.g. by an earlier reload of this module)
                _debugpy_listening = True
            print("WAITING FOR DEBUGGER ATTACH")
            # debugpy.wait_for_client()
            if not debugpy.is_client_connected():
                debugpy.wait_for_client()
            debugpy.breakpoint()
            print('break on this line')
        except:
            pass