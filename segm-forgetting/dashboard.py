from pathlib import Path

import gradio as gr
import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm
import plotly.express as px

from file import ROOT_PATH
from util import construct_dataset, construct_scenario, construct_model, construct_loader, N_CLASSES

CACHE_PATH = ROOT_PATH / "experiments/dashboard"


def save_plotly_figure(fig, out_path: Path, title: str):
    image_path = out_path / f"{title}.png"
    html_path = out_path / f"{title}.html"

    fig.write_image(str(image_path))
    fig.write_html(str(html_path))

    return image_path, html_path


def histogram_image(counts: torch.Tensor, title: str, out_path: Path):
    data = pd.DataFrame({
        "Labels": list(range(len(counts))),
        "Counts": counts.tolist(),
        "Colors": list(range(len(counts)))
    })
    fig = px.bar(data, x="Labels", y="Counts", color="Colors", title=title)
    fig.update_yaxes(type="log", title="Counts (Log Scale)")
    return save_plotly_figure(fig, out_path, title)


def class_distributions(scenario, out_path: Path):
    per_task_label_counts = []
    for task_id, task_set in enumerate(scenario):
        task_label_counts = torch.zeros(N_CLASSES)
        loader = construct_loader(task_set, batch_size=8, shuffle=False)
        for _, label, _ in tqdm(loader, desc=f"Aggregating task labels for task {task_id}"):
            task_label_counts += torch.bincount(label.flatten(), minlength=N_CLASSES)
        per_task_label_counts.append(task_label_counts)
        torch.save(per_task_label_counts, out_path)

    return per_task_label_counts


def get_class_dists(scenario, out_path):
    if not out_path.exists():
        train_task_label_counts = class_distributions(scenario, out_path)
    else:
        print(f"Loading class dists from {out_path}")
        train_task_label_counts = torch.load(out_path)
    dataset_label_counts = torch.sum(torch.stack(train_task_label_counts), dim=0)
    return dataset_label_counts, train_task_label_counts


def run_dashboard(
        dataset: DictConfig,
        model: DictConfig
):
    # Construct datasets and scenarios
    train_dataset = construct_dataset(dataset, train=True)
    val_dataset = construct_dataset(dataset, train=False)

    train_scenario, train_tasks_classes = construct_scenario(train_dataset)
    val_scenario, val_tasks_classes = construct_scenario(val_dataset)

    train_label_counts, train_per_task_label_counts = get_class_dists(train_scenario, CACHE_PATH / "train_counts.pt")
    val_label_counts, val_per_task_label_counts = get_class_dists(val_scenario, CACHE_PATH / "val_counts.pt")

    # Generate and save all histograms outside the Gradio block
    train_images = []
    for idx, task_label_counts in enumerate(train_per_task_label_counts):
        img_path, html_path = histogram_image(
            task_label_counts, f"Class_label_counts_task_{idx}", CACHE_PATH
        )
        train_images.append((idx, img_path))

    val_images = []
    for idx, task_label_counts in enumerate(val_per_task_label_counts):
        img_path, html_path = histogram_image(
            task_label_counts, f"Class_label_counts_task_{idx}_val", CACHE_PATH
        )
        val_images.append((idx, img_path))

    # Build the Gradio interface after figures are saved
    with gr.Blocks() as demo:
        with gr.Tab("Dataset"):
            with gr.Accordion("Train Label Counts"):
                with gr.Row():
                    # Load pre-generated images
                    for idx, img_path in train_images:
                        gr.Image(str(img_path), label=f"Task {idx} Counts", min_width=400)

            with gr.Accordion("Validation Label Counts"):
                with gr.Row():
                    # Load pre-generated images
                    for idx, img_path in val_images:
                        gr.Image(str(img_path), label=f"Task {idx} Validation Counts", min_width=400)

        with gr.Tab("Models"):
            model_obj = construct_model(model)
            # Further model-related components can be added here
            pass

        with gr.Tab("Analysis"):
            # Further analysis components can be added here
            pass

    demo.launch()


@hydra.main(version_base=None, config_path="../config", config_name="dashboard")
def main(cfg: DictConfig) -> None:
    run_dashboard(**cfg)


if __name__ == "__main__":
    main()
