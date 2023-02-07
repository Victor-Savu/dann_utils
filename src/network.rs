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

fn index(kind: &dann::NodeKind, nn: &dann::Dann<f64>) -> usize {
    match kind {
        dann::NodeKind::Input(id) => *id,
        dann::NodeKind::Output(id) => *id,
        dann::NodeKind::Latent(id) => {
            nn.latent.iter().position(|i| i == id).unwrap() + nn.inputs + nn.outputs
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

    pub fn from_dann(nn: &dann::Dann<f64>) -> Self {
        Self::new(
            (0..nn.inputs)
                .map(|id| {
                    let bias = nn.nodes[&dann::NodeKind::Input(id)].bias;
                    Node {
                        x: 0.,
                        y: 1. - id as f64 / (nn.inputs - 1) as f64,
                        id,
                        bias,
                    }
                })
                .chain((0..nn.outputs).map(|idx| {
                    let id = idx + nn.inputs;
                    let bias = nn.nodes[&dann::NodeKind::Output(id)].bias;
                    Node {
                        x: 1.,
                        y: 1.
                            - if nn.outputs == 1 {
                                0.5
                            } else {
                                idx as f64 / (nn.outputs - 1) as f64
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
        use plotly::{
            color::NamedColor,
            common::{HoverInfo, Mode},
            layout::{Axis, ShapeLayer, ShapeType},
            Plot, Scatter,
        };

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
