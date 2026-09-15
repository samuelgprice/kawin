# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:26:45 2026

@author: samth
"""

def debugInPlace():
    import os
    import sys

    is_jupyter = "ipykernel" in sys.modules
    if is_jupyter:
        try:
            import debugpy
            # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
            debugpy.listen(5678)
            print("WAITING FOR DEBUGGER ATTACH")
            # debugpy.wait_for_client()
            if not debugpy.is_client_connected():
                debugpy.wait_for_client()
            debugpy.breakpoint()
            print('break on this line')
        except:
            pass