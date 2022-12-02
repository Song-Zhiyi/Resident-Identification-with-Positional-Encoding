#!/usr/bin/env python3
if __name__ == "__main__":
    import icecream
    import logging
    #logging.basicConfig(level="DEBUG")
    from exp_run.cli import entry
    icecream.ic.disable()
    entry()