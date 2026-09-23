"""Draw the proposed classification design and the verified experimental scorer.

This produces documentation only. Dashed modules are proposed additions, not
claims about the completed BOP pilot. No perception or inference code is run.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/classification_framework_20260923"
INK = "#193047"
MUTED = "#53677b"
LINE = "#687c8e"
BLUE = "#eaf2fb"
TEAL = "#e9f5f1"
AMBER = "#fff4df"
GRAY = "#f2f5f8"

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                     "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})


def sheet(height, title, subtitle):
    fig = plt.figure(figsize=(18, height / 100), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1800), ylim=(height, 0))
    ax.axis("off")
    text(ax, 45, 34, title, 25, weight="bold")
    text(ax, 45, 83, subtitle, 13, color=MUTED)
    return fig, ax


def text(ax, x, y, value, size=14, color=INK, weight="normal", align="left"):
    return ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                   ha=align, va="top", linespacing=1.5, zorder=4)


def box(ax, x, y, width, height, title, lines=(), fill=GRAY, proposed=False,
        title_size=14, body_size=12.5):
    patch = FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0,rounding_size=10",
                           facecolor=fill, edgecolor="#b5c4d0", linewidth=1.4,
                           linestyle=(0, (5, 4)) if proposed else "solid", zorder=2)
    ax.add_patch(patch)
    text(ax, x + 18, y + 17, title, title_size, weight="bold")
    if lines:
        text(ax, x + 18, y + 57, "\n".join(lines), body_size)
    return patch


def arrow(ax, points, dashed=False, color=LINE):
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>", mutation_scale=14,
                                linewidth=1.5, color=color, zorder=3,
                                linestyle=(0, (5, 4)) if dashed else "solid"))


def save(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Catch labels spilling outside the figure before producing the files.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for ax in fig.axes:
        for label in ax.texts:
            extent = label.get_window_extent(renderer)
            assert bounds.contains(extent.x0, extent.y0) and bounds.contains(extent.x1, extent.y1), label.get_text()
    for extension in ("png", "svg"):
        fig.savefig(OUTPUT / f"{name}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def overview():
    fig, ax = sheet(1080, "CAD-guided object classification",
                    "Proposed SAM3 + depth-verification design, connected to our existing three-score matcher")
    text(ax, 45, 127, "REFERENCE PREPARATION  |  once per CAD library", 11.5, color=MUTED, weight="bold")
    box(ax, 45, 160, 365, 135, "Supported CAD library", [
        "All supported CAD identities", "Original metric scale"], GRAY)
    box(ax, 465, 160, 515, 135, "42 reference views per CAD", [
        "Rendered RGB + foreground masks", "Known rotations + original CAD geometry"], BLUE)
    box(ax, 1035, 160, 720, 135, "Cached reference bank", [
        "Normalized DINO CLS and foreground patch features", "Metric mesh retained for camera-view surface rendering"], BLUE)
    arrow(ax, [(410, 227), (465, 227)])
    arrow(ax, [(980, 227), (1035, 227)])

    text(ax, 45, 327, "RUNTIME  |  one calibrated RGB-D image", 11.5, color=MUTED, weight="bold")
    box(ax, 45, 430, 255, 180, "RGB-D observation", [
        "RGB image I", "Depth D + intrinsics K", "Estimated support plane"], GRAY)
    box(ax, 355, 430, 270, 180, "SAM3 proposals", [
        "Candidate masks {Mᵢ}", "Declared prompt policy", "Identity still unknown"], BLUE)
    box(ax, 680, 430, 320, 180, "Depth-check the masks", [
        "Valid depth / 3D surface support", "Nested-mask consistency", "Keep, reject, or flag"], TEAL, proposed=True)
    box(ax, 1055, 430, 355, 180, "Every mask × every CAD", [
        "Global G + patch P(v)", "Visible-surface geometry H(v)", "Fused score matrix Sᵢⱼ"], BLUE)
    box(ax, 1465, 430, 290, 180, "Rank and resolve", [
        "CAD query → rank regions", "Benchmark → classify regions", "Duplicates / unknowns"], AMBER,
        proposed=True, body_size=11.5)
    for first, second in [(300, 355), (625, 680), (1000, 1055), (1410, 1465)]:
        arrow(ax, [(first, 520), (second, 520)])
    text(ax, 327, 488, "RGB", 10, color=MUTED, align="center")
    arrow(ax, [(172, 430), (172, 375), (840, 375), (840, 430)])
    text(ax, 495, 348, "Depth + calibration + support-plane evidence", 11.5, color=MUTED, align="center")
    arrow(ax, [(1232, 295), (1232, 430)])
    text(ax, 1250, 358, "CAD references", 11, color=MUTED)

    box(ax, 355, 735, 645, 135, "Optional VLM-assisted mask review", [
        "Interpret ambiguous overlap or part-versus-whole masks", "Suggest point / box prompts; SAM3 generates revised masks"],
        AMBER, proposed=True, body_size=12)
    arrow(ax, [(840, 610), (840, 735)], dashed=True)
    text(ax, 858, 655, "Ambiguous masks", 11, color=MUTED)
    arrow(ax, [(490, 735), (490, 610)], dashed=True)
    text(ax, 472, 655, "Re-prompt", 11, color=MUTED, align="right")
    box(ax, 1465, 735, 290, 135, "Association output", [
        "Mask ID + CAD ID + score", "Winning view or review status"], TEAL, body_size=11.5)
    arrow(ax, [(1610, 610), (1610, 735)])
    text(ax, 1060, 735, "Solid: available components", 11.5, color=MUTED)
    text(ax, 1060, 776, "Dashed: proposed additions\nor policies to validate", 11.5, color=MUTED)

    text(ax, 45, 931, "Depth verifies SAM masks; it does not add separate depth-generated masks in this proposed design.", 12)
    text(ax, 45, 969, "CAD and mask counts need not match. Repeated CAD instances are allowed; there is no forced one-to-one assignment.", 12)
    text(ax, 45, 1007, "Benchmark inputs exclude ground-truth identities, masks and counts. Pose refinement and robot actions are outside this figure.", 12)
    save(fig, "classification_overview")


def scorer():
    fig, ax = sheet(1410, "Inside the CAD-to-observation matcher",
                    "Current tested scoring baseline — view selection, table rejection and fusion weights remain under study")
    box(ax, 45, 137, 1710, 130, "One observed mask i × one CAD j", [
        "Observed masked RGB and CAD templates: white background, 224 × 224 aspect-preserving letterbox",
        "DINOv2-L/14 → normalized CLS + 16 × 16 patch tokens | retain all 42 CAD-view scores"], BLUE, body_size=13)
    arrow(ax, [(367, 267), (367, 325)])
    arrow(ax, [(1255, 267), (1255, 325)])

    box(ax, 45, 325, 645, 208, "1A  Global similarity", fill=BLUE)
    text(ax, 65, 383, r"$c_{ij}(v)=\mathrm{clip}_{[0,1]}\!\left(\cos(f_i^{\rm CLS},f_{jv}^{\rm CLS})\right)$", 17)
    text(ax, 65, 438, r"$G_{ij}=\frac{1}{5}\sum_{v\in\mathrm{Top5}_{c}}c_{ij}(v)$", 20)
    text(ax, 65, 501, "One top-five average per CAD–region pair", 11.5, color=MUTED)
    box(ax, 755, 325, 1000, 208, "1B  Patch similarity for every view", [
        "Compute Pᵢⱼ(v) independently for all 42 CAD templates.",
        "Each observed foreground patch finds its best CAD-patch cosine match.",
        "Average the matches; foreground gate uses mask occupancy > 0.5.",
        "Global top-five averaging does not restrict these patch comparisons."], BLUE, body_size=13)

    box(ax, 755, 595, 1000, 177, "2  Select up to five admissible views", [
        "Visit views in descending patch-score order.",
        "Template rotation + fitted translation → table / camera checks.",
        "Retain the first five passing views, or fewer if fewer are available."], GRAY, body_size=13)
    arrow(ax, [(1255, 533), (1255, 595)])
    text(ax, 45, 1044, "Pose errors can make the table check reject the correct CAD.", 11, color=MUTED)

    box(ax, 755, 847, 1000, 177, "3  Verify the rendered visible surface", [
        "Render CAD depth and support at the same view v and fitted pose.",
        "Compare observed-only pixels, CAD-only pixels and shared-pixel depth.",
        "Use native-resolution measured depth, physical scale and visibility handling."], TEAL, body_size=13)
    arrow(ax, [(1255, 772), (1255, 847)])

    box(ax, 45, 595, 625, 429, "Visible-surface score H(v)", fill=TEAL)
    text(ax, 65, 647, r"$H(v)=1-\frac{p_{\rm obs}+p_{\rm cad}+p_{\rm depth}}{3}$", 20)
    text(ax, 65, 713, r"$p_{\rm obs}=\frac{|\Omega_o\setminus\Omega_c|}{|\Omega_o\cup\Omega_c|}$"
         + "     " + r"$p_{\rm cad}=\frac{|\Omega_c\setminus\Omega_o|}{|\Omega_o\cup\Omega_c|}$", 17)
    text(ax, 65, 801, r"$p_{\rm depth}=\frac{1}{|\mathcal{I}|}\sum_{u\in\mathcal{I}}\min\!\left(\frac{|D_o(u)-D_c(u)|}{\sigma_d},1\right)$", 16)
    text(ax, 65, 897, r"$\mathcal{I}=\Omega_o\cap\Omega_c,\qquad \sigma_d=0.005\,\mathrm{m}$", 17)
    text(ax, 65, 940, "Ωo: valid observed object pixels\nΩc: assessable visible CAD support\nNo valid overlap → geometry unassessable", 11.5, color=MUTED)

    box(ax, 755, 1097, 1000, 183, "4  Fuse scores, then select the winning view", fill=BLUE)
    text(ax, 777, 1152, r"$F_{ij}(v)=0.25\,G_{ij}+0.25\,P_{ij}(v)+0.50\,H_{ij}(v)$", 21)
    text(ax, 777, 1210, r"$S_{ij}=\max_{v\in\mathcal{V}_{ij}}F_{ij}(v)$"
         + "     " + r"$v^*_{ij}=\arg\max_{v\in\mathcal{V}_{ij}}F_{ij}(v)$", 20)
    arrow(ax, [(1255, 1024), (1255, 1097)])
    arrow(ax, [(367, 533), (367, 560), (714, 560), (714, 1188), (755, 1188)])
    text(ax, 731, 1077, "G", 13, color=MUTED)
    box(ax, 45, 1097, 625, 183, "Interpretation", [
        "Repeat for every mask–CAD pair to form S.",
        "Column ranking: find an object for a requested CAD.",
        "Row ranking: classify each observed region.",
        "Winning view is an initialization, not a refined pose."], GRAY, body_size=12)
    text(ax, 45, 1320, "Current fitting still uses 5 mm sampled clouds and translation-only iterations. H itself uses image masks and depth, not ICP fitness.", 11.5)
    text(ax, 45, 1356, "The 0.005 m depth-penalty scale and the 5 mm fitting voxel size are separate settings. The displayed fusion weights are the fixed baseline.", 11.5)
    save(fig, "classification_scoring")


if __name__ == "__main__":
    overview()
    scorer()
    print(OUTPUT)
