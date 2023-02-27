use itertools_num::linspace;
use plotly::{
    color::NamedColor,
    common::{ColorScale, ColorScaleElement, HoverInfo, Mode},
    layout::{Axis, GridPattern, LayoutGrid, ShapeLayer, ShapeType},
    Histogram, Plot, Scatter,
};

#[derive(Clone, Copy, Debug)]
pub struct Node {
    pub x: f64,
    pub y: f64,
    pub id: usize,
    pub bias: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Edge {
    source: Node,
    target: Node,
    weight: f64,
}

#[derive(Clone, Debug)]
pub struct Network {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

fn index<const INPUTS: usize, const OUTPUTS: usize>(kind: &dann::NodeKind, nn: &dann::Dann<f64, INPUTS, OUTPUTS>) -> usize {
    match kind {
        dann::NodeKind::Input(id) => *id,
        dann::NodeKind::Output(id) => *id,
        dann::NodeKind::Latent(id) => {
            nn.latent.iter().position(|i| i == id).unwrap() + INPUTS + OUTPUTS
        }
    }
}

impl Network {
    pub fn new(nodes: Vec<Node>, edges: Vec<(usize, usize, f64)>) -> Self {
        Self {
            edges: edges
                .into_iter()
                .map(|(s, t, weight)| Edge {
                    source: nodes[s],
                    target: nodes[t],
                    weight,
                })
                .collect(),
            nodes,
        }
    }

    pub fn from_dann<const INPUTS: usize, const OUTPUTS: usize>(nn: &dann::Dann<f64, INPUTS, OUTPUTS>) -> Self {
        Self::new(
            (0..INPUTS)
                .map(|id| {
                    let bias = nn.nodes[&dann::NodeKind::Input(id)].bias;
                    Node {
                        x: 0.,
                        y: 1. - id as f64 / (INPUTS - 1) as f64,
                        id,
                        bias,
                    }
                })
                .chain((0..OUTPUTS).map(|idx| {
                    let id = idx + INPUTS;
                    let bias = nn.nodes[&dann::NodeKind::Output(id)].bias;
                    Node {
                        x: 1.,
                        y: 1.
                            - if OUTPUTS == 1 {
                                0.5
                            } else {
                                idx as f64 / (OUTPUTS - 1) as f64
                            },
                        id,
                        bias,
                    }
                }))
                .chain(nn.latent.iter().enumerate().map(|(idx, &id)| {
                    let bias = nn.nodes[&dann::NodeKind::Latent(id)].bias;
                    let p = (idx + 1) as f64 / (nn.latent.len() + 1) as f64; // curve parameter [1/(len+1)..len/(len+1)]
                    let a = (p - 1.) * std::f64::consts::PI / 2.; // angle on curve
                    Node {
                        x: a.cos(),
                        y: a.sin() + 1.,
                        id,
                        bias,
                    }
                }))
                .collect(),
            nn.nodes
                .iter()
                .flat_map(|(skind, snode)| {
                    let sid = index(skind, nn);
                    snode
                        .weights
                        .iter()
                        .map(move |(tkind, &weight)| (sid, index(tkind, nn), weight))
                })
                .collect(),
        )
    }

    pub fn evcxr_display(&self) {
        let nodes = plotly::Scatter::new(
            self.nodes.iter().map(|&Node { x, .. }| x).collect(),
            self.nodes.iter().map(|&Node { y, .. }| y).collect(),
        )
        .mode(Mode::Markers)
        .show_legend(false)
        .hover_text_array(
            self.nodes
                .iter()
                .map(|&Node { id, bias, .. }| format!("{id}: {bias:.3}"))
                .collect(),
        )
        .hover_info(HoverInfo::Text);

        let weights = Scatter::new(
            self.edges
                .iter()
                .map(|Edge { source, target, .. }| (source.x + target.x) / 2.)
                .collect(),
            self.edges
                .iter()
                .map(|Edge { source, target, .. }| (source.y + target.y) / 2.)
                .collect(),
        )
        .mode(Mode::Text)
        .text_array(
            self.edges
                .iter()
                .map(|Edge { weight, .. }| format!("{weight:.3}"))
                .collect(),
        )
        .show_legend(false);

        let mut layout = plotly::Layout::new()
            .x_axis(
                Axis::new()
                    .zero_line(false)
                    .show_grid(false)
                    .show_tick_labels(false),
            )
            .y_axis(
                Axis::new()
                    .zero_line(false)
                    .show_grid(false)
                    .show_tick_labels(false),
            );

        for Edge {
            source,
            target,
            weight,
        } in &self.edges
        {
            layout.add_shape(
                plotly::layout::Shape::new()
                    .opacity(0.7)
                    .shape_type(ShapeType::Line)
                    .x0(source.x)
                    .y0(source.y)
                    .x1(target.x)
                    .y1(target.y)
                    .line(
                        plotly::layout::ShapeLine::new()
                            .color(if *weight < 0. {
                                NamedColor::Orange
                            } else {
                                NamedColor::Blue
                            })
                            .width(1. + weight.abs()),
                    )
                    .layer(ShapeLayer::Below),
            )
        }

        let mut plot = Plot::new();
        plot.set_layout(layout);
        plot.add_trace(nodes);
        plot.add_trace(weights);
        plot.evcxr_display();
    }
}

pub fn hist_weights<'a, T: 'a>(
    nodes: impl IntoIterator<Item = (&'a T, &'a dann::Node<f64>)> + 'a,
    bins: usize,
) -> Plot {
    let mut plot = Plot::new();
    plot.add_trace(
        Histogram::new(
            nodes
                .into_iter()
                .flat_map(|(_, node)| node.weights.values().cloned())
                .collect(),
        )
        .n_bins_x(bins),
    );
    plot
}

#[derive(Debug)]
pub struct Gridsize {
    inputs: usize,
    outputs: usize,
    latent_cols: usize,
    latent: std::collections::HashMap<usize, usize>,
}

impl Gridsize {
    pub fn new<const INPUTS: usize, const OUTPUTS: usize>(nn: &dann::Dann<f64, INPUTS, OUTPUTS>) -> Self {
        let rows = INPUTS.max(OUTPUTS);
        let latent_cols = (nn.latent.len() + rows - 1) / rows;
        Self {
            inputs: INPUTS,
            outputs: OUTPUTS,
            latent_cols,
            latent: nn
                .latent
                .iter()
                .enumerate()
                .map(|(idx, id)| {
                    let row = idx / latent_cols;
                    let col = idx % latent_cols;
                    (*id, 1 + row * (latent_cols + 2) + (col + 1))
                })
                .collect(),
        }
    }

    pub fn subplot(&self, id: &dann::NodeKind) -> usize {
        use dann::NodeKind::*;
        match id {
            Input(i) => *i * self.cols() + 1,
            Latent(i) => self.latent[i],
            Output(i) => {
                (*i - self.inputs) * self.cols()
                    + self.cols()
                    + (self.rows() - self.outputs) / 2 * self.cols()
            }
        }
    }

    pub fn rows(&self) -> usize {
        self.inputs.max(self.outputs)
    }

    pub fn cols(&self) -> usize {
        self.latent_cols + 2
    }
}

pub fn show<'a, H>(gs: &Gridsize, hm: H, inputs: Box<plotly::Scatter<f64, f64>>) -> Plot
where
    H: IntoIterator<Item = (&'a dann::NodeKind, &'a Vec<Vec<f64>>)> + 'a,
{
    let mut plot = Plot::new();

    for (id, values) in hm {
        plot.add_trace(
            plotly::Contour::new(
                linspace(-1.2, 1.2, 100).collect(),
                linspace(-1.2, 1.2, 100).collect(),
                values.clone(),
            )
            .opacity(0.38)
            .name(&format!("{id:?}"))
            .x_axis(&format!("x{}", gs.subplot(id)))
            .y_axis(&format!("y{}", gs.subplot(id)))
            .show_scale(false)
            .show_legend(false)
            .color_scale(ColorScale::Vector(vec![
                ColorScaleElement(-1., "#0000ff".to_string()),
                ColorScaleElement(0., "#000000".to_string()),
                ColorScaleElement(1., "#ff00ff".to_string()),
            ])),
        );
        plot.add_trace(
            inputs
                .clone()
                .x_axis(format!("x{}", gs.subplot(id)))
                .y_axis(format!("y{}", gs.subplot(id)))
                .hover_info(HoverInfo::Skip),
        );
    }

    plot.set_layout(
        plotly::layout::Layout::new()
            .width((1200 as f64 * (gs.cols() as f64) / (gs.rows() as f64)) as usize)
            .height(1200)
            .grid(
                LayoutGrid::new()
                    .rows(gs.rows())
                    .columns(gs.cols())
                    .pattern(GridPattern::Independent),
            ),
    );
    plot
}
