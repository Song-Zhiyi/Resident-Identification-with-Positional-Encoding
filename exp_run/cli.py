from pprint import pprint
import click
@click.command()
@click.argument("presets_to_run", metavar="PRESETS", nargs=-1)
@click.option("--on-error", type=click.Choice(["stop", "skip"]), default="skip")
@click.option("--exp-name", type=str, default=None)
@click.option("--repeat", type=int, default=5)
def entry(exp_name: str, repeat: int, presets_to_run: list[str], on_error: str):
    presets_to_run = frozenset(presets_to_run)
    skip_error = (on_error == "skip")

    if exp_name is None:
        import secrets
        exp_name = secrets.token_hex(8)

    from . import lib
    presets = lib.load_preset()

    if len(presets_to_run) == 0:
        pprint(list(presets.keys()))
        return

    from tqdm import tqdm
    progress = tqdm(total=len(presets_to_run), unit="preset", leave=False)
    for preset in presets_to_run:
        try:
            progress.set_description(preset)
            lib.start_preset(
                presets[preset].build_preset(exp_name=exp_name, repeat=repeat)
            )
            progress.update()
        except Exception as e:
            if not skip_error:
                raise