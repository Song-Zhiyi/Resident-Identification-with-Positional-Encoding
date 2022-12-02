import click
from PIL import Image
from sys import exit
import sys
import numpy as np
import multiprocessing as mp
from pprint import pprint
import pickle

class ImageShowingProcess(mp.Process):
    def __init__(self, image_path, scale):
        super().__init__()
        self.queue = mp.Queue()
        self.image_path = image_path
        self.scale = scale

    def run(self):
        #import os
        #os.close(0)
        #os.close(1)
        #os.close(2)
        #sys.stdin = open(os.devnull, "w")
        #sys.stdout = open(os.devnull, "w")
        #sys.stderr = open(os.devnull, "w")
        self._run()

    def _run(self):
        import cv2

        image_path = self.image_path
        scale = self.scale

        def on_mouse_click(event, x, y, flags, param):
            if event == 1:
                self.queue.put((x, y))
                self.img = cv2.circle(self.img, (x, y), radius=5, color=(0,255,0))
                cv2.imshow(image_path, self.img)

        cv2.namedWindow(image_path)
        cv2.setMouseCallback(image_path, on_mouse_click)
        self.img = cv2.imread(image_path)
        y, x, _ = self.img.shape
        x = int(x * scale)
        y = int(y * scale)
        self.img = cv2.resize(self.img, (x, y))
        cv2.imshow(image_path, self.img)

        while True:
            cv2.waitKey()

class Work:
    def __init__(self, image_path, scale, save_path):
        self.p = ImageShowingProcess(image_path, scale)

        self.sensor_map = {}
        self.wall_map = {}
        self.base = None
        self.save_path = save_path

    def save(self, path):
        with open(path, "wb") as fp:
            to_save = {
                'sensor_map': self.sensor_map,
                'wall_map': self.wall_map,
                'base': self.base,
            }
            pickle.dump(to_save, fp)
    def load(self, path):
        with open(path, "rb") as fp:
            loaded = pickle.load(fp)
            self.sensor_map = loaded['sensor_map']
            self.wall_map = loaded['wall_map']
            self.base = loaded['base']

    def status(self):
        pprint({
            'sensor_map': self.sensor_map,
            'wall_map': self.wall_map,
            'base': self.base,
        })

    def start(self):
        try:
            self._start()
        finally:
            try:
                self.p.terminate()
            except:
                pass

            if self.save_path is not None:
                self.save(self.save_path)

    def _start(self):
        self.p.start()
        try:
            self._simple_cmd()
        except (EOFError, KeyboardInterrupt):
            pass

    def clear_queue(self):
        try:
            while True:
                self.p.queue.get_nowait()
        except Exception:
            pass

    def get_coord(self):
        return self.p.queue.get()

    def _exec_command(self, command):
        parts = command.split(" ")
        command = parts[0]
        args = parts[1:]

        self.clear_queue()

        if command == "s":
            sname = args[0]
            coord = self.get_coord()
            self.sensor_map[sname] = coord
            print(f"add sensor {sname} {coord}")
        elif command == "rs":
            del self.sensor_map[args[0]]
            print(f"remove sensor {args[0]}")
        elif command == "ls":
            pprint(self.sensor_map)
        elif command == "b":
            coord = self.get_coord()
            self.base = coord
            print(f"base set to {coord}")
        elif command == "rb":
            removed = self.base
            self.base = None
            print(f"removed base point {removed}")
        elif command == "lb":
            print(self.base)
        elif command == "w":
            wname = args[0]
            not_orth = (len(args) >= 2 and args[1] == "nl")

            x1, y1 = self.get_coord()
            x2, y2 = self.get_coord()

            if not not_orth:
                if abs(x1 - x2) < 10:
                    x1 = x2
                elif abs(y1 - y2) < 10:
                    y1 = y2

            c1 = (x1, y1)
            c2 = (x2, y2)

            self.wall_map[wname] = c1, c2
            print(f"add wall {c1} {c2}")
        elif command == "rw":
            del self.wall_map[args[0]]
        elif command == "save":
            to_save = args[0] if len(args) >= 1 else self.save_path
            self.save(to_save)
            print(f"saved to {to_save}")
        elif command == "load":
            to_load = args[0] if len(args) >= 1 else self.save_path
            self.load(to_load)
            print(f"loaded from {to_load}")
        elif command == "status":
            self.status()
        elif command == "quit":
            exit(0)

    def _simple_cmd(self):
        while True:
            command = input("> ")
            try:
                self._exec_command(command)
            except Exception as e:
                print(type(e).__name__, e)

@click.command()
@click.argument(
    "image_path",
    type=click.Path(exists=True, dir_okay=False,
                    resolve_path=True, readable=True),
    metavar="<path to image>")
@click.option("--scale", type=float, default=1, help="File to save")
@click.option(
    "--save",
    type=click.Path(exists=False, dir_okay=False,
                    resolve_path=True, writable=True),
    default=None,
    help="scale of image to display")
def main(image_path, scale, save):
    w = Work(image_path, scale, save)
    w.start()

if __name__ == '__main__':
    main()
