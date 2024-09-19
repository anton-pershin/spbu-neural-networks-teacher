import matplotlib.pyplot as plt


class TbMatrixPlotSaverHook:
    def __init__(self, hook_manager, name):
        self.hook_manager = hook_manager
        self.name = name

    def __call__(self, module, args, output):
        if not self.hook_manager.enabled:
            return
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        conts = ax.imshow(output.detach().squeeze(0))
        fig.colorbar(conts)
        fig.tight_layout()
        self.hook_manager.tb_writer.add_figure(f"Outputs/{self.name}", fig, 0)       


class HookManager:
    def __init__(self, tb_writer):
        self.enabled = False
        self.tb_writer = tb_writer

    def emit_hook(self, name, hook_class):
        hook = hook_class(self, name)
        return hook

