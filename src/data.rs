pub trait Model {
    fn eval(&self, x: f64, y: f64) -> bool;
}

pub trait Sample {
    fn sample(&self, a: f64, b: f64) -> (f64, f64);
}

#[derive(Copy, Clone)]
pub struct Disk {
    radius: f64,
}

impl Disk {
    pub fn new(radius: f64) -> Self {
        Self{radius}
    }
}

impl Model for Disk {
    fn eval(&self, x: f64, y: f64) -> bool {
        x * x + y * y < self.radius * self.radius
    }
}

impl Sample for Disk {
    fn sample(&self, a: f64, b: f64) -> (f64, f64) {
        Doughnut::new(0., self.radius).sample(a, b)
    }
}

pub struct Doughnut {
    inner_radius: f64,
    outer_radius: f64,
}

impl Doughnut {
    pub fn new(inner_radius: f64, outer_radius: f64) -> Self {
        Self {
            inner_radius,
            outer_radius,
        }
    }
}

impl Model for Doughnut {
    fn eval(&self, x: f64, y: f64) -> bool {
        Disk::new(self.outer_radius).eval(x, y) & !Disk::new(self.inner_radius).eval(x, y)
    }
}

impl Sample for Doughnut {
    fn sample(&self, a: f64, b: f64) -> (f64, f64) {
        let osq = self.outer_radius * self.outer_radius;
        let isq = self.inner_radius * self.inner_radius;
        let r = (isq + a * (osq - isq)).sqrt();
        let t = b * 2. * std::f64::consts::PI;
        (r * t.cos(), r * t.sin())
    }
}