import matplotlib.pyplot as plt
import sys
import os
import numpy as np

# --- HIGH-PERFORMANCE JSON LOADING ---
try:
    import simdjson
    USE_SIMDJSON = True
except ImportError:
    import json
    USE_SIMDJSON = False
    print("⚠️ pysimdjson not found. Falling back to standard json.")

def load_json_fast(json_file_path):
    """Uses simdjson for C++ level parsing speeds, with a safe fallback."""
    if USE_SIMDJSON:
        parser = simdjson.Parser()
        with open(json_file_path, 'rb') as f:
            proxy = parser.parse(f.read())
            # Convert simdjson proxy to standard Python dict/lists for matplotlib compatibility
            return {k: list(v) if hasattr(v, '__iter__') and not isinstance(v, str) else v for k, v in proxy.items()}
    else:
        with open(json_file_path, 'r') as f:
            return json.load(f)

def update_metrics_plot(json_file_path, fig, axes):
    if not os.path.exists(json_file_path):
        return

    data = load_json_fast(json_file_path)

    train_step_loss = data.get('train_step_loss', [])
    train_epoch_loss = data.get('train_epoch_loss', [])
    val_epoch_loss = data.get('val_epoch_loss', [])
    val_task_losses = data.get('val_task_losses', [])

    if not train_step_loss and not train_epoch_loss:
        return

    has_epoch_data = len(train_epoch_loss) > 0
    has_task_data = len(val_task_losses) > 0
    
    # Clear old plots
    for ax in axes:
        ax.cla()

    # --- PLOT 1: Step-Level Loss (Real-Time) ---
    ax1 = axes[0]
    steps = range(1, len(train_step_loss) + 1)
    window = min(20, max(1, len(train_step_loss)))
    
    if window > 1 and len(train_step_loss) >= window:
        smoothed_loss = np.convolve(train_step_loss, np.ones(window)/window, mode='valid')
        smoothed_steps = range(window, window + len(smoothed_loss))
        ax1.plot(steps, train_step_loss, color='lightblue', alpha=0.4, label='Raw Step Loss')
        ax1.plot(smoothed_steps, smoothed_loss, color='blue', linewidth=2.5, label=f'Smoothed (Window={window})')
    else:
        ax1.plot(steps, train_step_loss, color='blue', linewidth=2.5, label='Step Loss')
        
    # Overlay Validation Step Loss if it exists
    val_step_loss = data.get('val_step_loss', [])
    if val_step_loss:
        val_steps = np.linspace(1, len(train_step_loss), len(val_step_loss))
        ax1.scatter(val_steps, val_step_loss, color='red', s=20, zorder=5, label='Val Loss (Checkpoints)')

    ax1.set_title('Real-Time Training Progress (Step-Level)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # --- PLOT 2: Epoch-Level Loss (The Overfitting Check) ---
    ax2 = axes[1]
    if has_epoch_data:
        epochs = range(1, len(train_epoch_loss) + 1)
        ax2.plot(epochs, train_epoch_loss, 'b-o', label='Global Training Loss', linewidth=2.5, markersize=8)
        if val_epoch_loss:
            val_epochs = range(1, len(val_epoch_loss) + 1)
            ax2.plot(val_epochs, val_epoch_loss, 'r-o', label='Global Validation Loss', linewidth=2.5, markersize=8)
        ax2.set_title('Epoch-Level Loss (Overfitting / Underfitting Analysis)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_xticks(list(epochs))
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "Waiting for Epoch 1 to finish...", ha='center', va='center', fontsize=12)
        ax2.set_title('Epoch-Level Loss', fontsize=14, fontweight='bold')

    # --- PLOT 3: Per-Task Validation Loss (The Interference Check) ---
    ax3 = axes[2]
    if has_task_data:
        task_names = list(val_task_losses[0].keys())
        epochs = range(1, len(val_task_losses) + 1)
        colors = {'CoMMU (Structure)': 'purple', 'EMOPIA (Emotion)': 'orange', 'SLakh (Orchestration)': 'green'}
        
        for task in task_names:
            task_vals = [d.get(task, None) for d in val_task_losses]
            valid_epochs = [e for e, v in zip(epochs, task_vals) if v is not None]
            valid_vals = [v for v in task_vals if v is not None]
            color = colors.get(task, 'gray')
            ax3.plot(valid_epochs, valid_vals, '-o', color=color, label=task, linewidth=2.5, markersize=8)
            
        ax3.set_title('Per-Task Validation Loss (Catastrophic Interference Check)', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_xticks(list(epochs))
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "Waiting for Validation to run...", ha='center', va='center', fontsize=12)
        ax3.set_title('Per-Task Validation Loss', fontsize=14, fontweight='bold')

    fig.tight_layout()
    output_image = json_file_path.replace('.json', '_analysis.png')
    fig.savefig(output_image, dpi=150, bbox_inches='tight')

    # --- TERMINAL DIAGNOSTICS ---
    print("\n" + "="*50)
    print("📊 MODEL HEALTH DIAGNOSTICS")
    print("="*50)
    if has_epoch_data and val_epoch_loss and len(train_epoch_loss) >= 2:
        train_trend = train_epoch_loss[-1] - train_epoch_loss[0]
        val_trend = val_epoch_loss[-1] - val_epoch_loss[0]
        print(f"Latest Global Train Loss: {train_epoch_loss[-1]:.4f}")
        print(f"Latest Global Val Loss:   {val_epoch_loss[-1]:.4f}\n")
        if train_trend < -0.05 and val_trend > 0.05:
            print("🔴 OVERFITTING DETECTED (Memorization)!")
        elif train_trend > -0.05 and val_trend > -0.05:
            print("🔴 UNDERFITTING DETECTED (Too Simple)!")
        elif train_trend < 0 and val_trend < 0:
            print("🟢 HEALTHY (The Goldilocks Zone)!")
        else:
            print("🟡 TRANSITIONAL PHASE")
            
        if has_task_data:
            print("\n--- Per-Task Latest Val Loss ---")
            for task, val in val_task_losses[-1].items():
                print(f"   {task}: {val:.4f}")
    elif has_epoch_data:
        print("⏳ Only Training Epoch data available so far.")
    else:
        print("⏳ Epoch 0 is still in progress.")
        if train_step_loss:
            print(f"   -> Current Step Loss: {train_step_loss[-1]:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        output_dir = "moonbeam_checkpoint/multi_task_lora"
        if os.path.exists(output_dir):
            json_files = [f for f in os.listdir(output_dir) if f.startswith("metrics_data") and f.endswith(".json")]
            if json_files:
                target_file = os.path.join(output_dir, max(json_files, key=lambda f: os.path.getmtime(os.path.join(output_dir, f))))
            else: target_file = None
        else: target_file = None

    if target_file:
        print(f"📊 Starting 3-Panel Dashboard for: {target_file}")
        plt.ion() 
        fig, axes = plt.subplots(3, 1, figsize=(10, 16)) 
        try:
            while True:
                update_metrics_plot(target_file, fig, axes) 
                plt.pause(10) 
        except KeyboardInterrupt:
            print("\n👋 Dynamic plotting stopped by user.")
            plt.ioff() 
            plt.show() 
    else:
        print("❌ Could not find metrics file.")