"""
CLI tool for classifying a URL as phishing or legitimate.
 
Usage:
    python -m src.predict predict https://example.com
    python -m src.predict predict https://suspicious.example --verbose
    python -m src.predict info
"""
 
from __future__ import annotations
 
import sys
from pathlib import Path
 
import joblib
import numpy as np
import pandas as pd
import typer
 
from src.features import features_for_model, extract_features
 
app = typer.Typer(
    help="Phishing URL detector — predict whether a URL is phishing or legitimate.",
    add_completion=False,
)
 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "phishing_model.pkl"
 
 
# ---------- Color helpers (no extra deps; standard ANSI) ----------
def _supports_color() -> bool:
    return sys.stdout.isatty()
 
 
def _c(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"
 
 
red    = lambda s: _c(s, "31;1")
green  = lambda s: _c(s, "32;1")
yellow = lambda s: _c(s, "33;1")
dim    = lambda s: _c(s, "2")
bold   = lambda s: _c(s, "1")
 
 
# ---------- Model loading ----------
def load_model(model_path: Path):
    if not model_path.exists():
        typer.echo(red(f"Error: model not found at {model_path}"))
        typer.echo("Run `python -m src.train` first, or run the Day 4 notebook to produce phishing_model.pkl.")
        raise typer.Exit(code=1)
    return joblib.load(model_path)
 
 
# ---------- Commands ----------
@app.command()
def predict(
    url: str = typer.Argument(..., help="The URL to classify"),
    model_path: Path = typer.Option(DEFAULT_MODEL_PATH, "--model", "-m", help="Path to trained model bundle"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show top contributing features"),
):
    """Classify a URL as phishing or legitimate."""
    bundle = load_model(model_path)
    model = bundle["model"]
    scaler = bundle.get("scaler")
    feature_names = bundle["feature_names"]
 
    # Extract features
    values, missing = features_for_model(url, feature_names)
    # Wrap in a DataFrame with the original feature names so scikit-learn
    # doesn't warn about missing feature names (the model was fit on a DataFrame).
    X = pd.DataFrame([values], columns=feature_names)
 
    # Scale if the bundle includes a scaler (Logistic Regression needs it; RF doesn't)
    if scaler is not None:
        X = scaler.transform(X)
 
    # Predict
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0]
    # In this dataset: 0 = phishing, 1 = legitimate
    phish_prob, legit_prob = float(proba[0]), float(proba[1])
 
    # ---------- Output ----------
    typer.echo("")
    typer.echo(bold("URL: ") + url)
 
    if pred == 0:
        label = red("PHISHING")
        confidence = phish_prob
    else:
        label = green("LEGITIMATE")
        confidence = legit_prob
 
    typer.echo(bold("Prediction: ") + label + dim(f"  (confidence: {confidence:.1%})"))
    typer.echo(dim(f"  Probability legit: {legit_prob:.1%}   phishing: {phish_prob:.1%}"))
 
    if missing:
        typer.echo("")
        typer.echo(yellow(
            f"⚠  {len(missing)} feature(s) the model expects could not be extracted from the URL string alone."
        ))
        typer.echo(dim(
            "   These were filled with 0 and may reduce prediction confidence. "
            "See README → Limitations."
        ))
 
    if verbose:
        _print_verbose_breakdown(url, model, feature_names, values)
 
 
def _print_verbose_breakdown(url: str, model, feature_names: list[str], values: list[float]) -> None:
    """Show the most influential features for this specific prediction."""
    typer.echo("")
    typer.echo(bold("Top contributing features for this URL:"))
 
    # Use the model's global feature importances (Random Forest) or coefficients
    # (Logistic Regression). Then multiply by the actual feature value to get a
    # per-prediction influence score.
    if hasattr(model, "feature_importances_"):
        global_importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        global_importance = np.abs(model.coef_[0])
    else:
        typer.echo(dim("  (Model does not expose feature importances.)"))
        return
 
    influence = np.array(values) * global_importance
    top_idx = np.argsort(influence)[::-1][:5]
 
    extracted = extract_features(url)
 
    for i in top_idx:
        name = feature_names[i]
        value = values[i]
        # Format the value nicely depending on size
        if name in extracted:
            val_str = f"{value:.3f}" if isinstance(value, float) and not value.is_integer() else f"{int(value)}"
        else:
            val_str = dim("not extracted")
        typer.echo(f"  • {name:<32} = {val_str}")
 
 
@app.command()
def info(
    model_path: Path = typer.Option(DEFAULT_MODEL_PATH, "--model", "-m"),
):
    """Print metadata about the currently saved model."""
    bundle = load_model(model_path)
    typer.echo("")
    typer.echo(bold("Model bundle: ") + str(model_path.relative_to(PROJECT_ROOT)))
    typer.echo(bold("Model type: ")   + bundle.get("model_type", "unknown"))
    typer.echo(bold("Feature count: ")+ str(len(bundle["feature_names"])))
    typer.echo(bold("Scaler: ")       + ("included" if bundle.get("scaler") else "none"))
 
    metrics = bundle.get("metrics")
    if metrics:
        typer.echo("")
        typer.echo(bold("Test-set metrics:"))
        for k, v in metrics.items():
            typer.echo(f"  {k:<10} {v:.4f}")
 
 
if __name__ == "__main__":
    app()
 