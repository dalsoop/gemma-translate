use crate::app::App;
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Gauge, Paragraph},
    Frame,
};

pub fn draw(f: &mut Frame, area: Rect, app: &App) {
    let ep_count = app.cfg.api_endpoints.len();
    let block = Block::default()
        .title(format!(" GPU ({} endpoints) ", ep_count))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let gpus = &app.gpu.gpus;
    if gpus.is_empty() {
        let msg = app.gpu.error.clone().unwrap_or_else(|| "폴링 대기…".into());
        f.render_widget(Paragraph::new(msg).style(Style::default().fg(Color::DarkGray)), inner);
        return;
    }

    let row_h = 2u16;
    for (i, g) in gpus.iter().enumerate() {
        let y = inner.y + i as u16 * row_h;
        if y + 1 >= inner.y + inner.height { break; }
        let label_rect = Rect { x: inner.x, y, width: inner.width, height: 1 };
        let gauge_rect = Rect { x: inner.x, y: y + 1, width: inner.width, height: 1 };

        // endpoint 매핑 표시 (포트 8080+i)
        let port_hint = if (i as usize) < ep_count {
            let ep = &app.cfg.api_endpoints[i as usize];
            let port = ep.rsplit(':').next().unwrap_or("?");
            format!(" :{port}")
        } else {
            String::new()
        };

        let label = Line::from(vec![
            Span::styled(
                format!("GPU{}{} ", g.index, port_hint),
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("{:>3}%  ", g.util_pct)),
            Span::raw(format!("{:>5}/{} MiB  ", g.mem_used_mib, g.mem_total_mib)),
            Span::styled(
                format!("{}°C", g.temp_c),
                Style::default().fg(if g.temp_c > 80 { Color::Red }
                    else if g.temp_c > 65 { Color::Yellow }
                    else { Color::Green }),
            ),
        ]);
        f.render_widget(Paragraph::new(label), label_rect);

        let mem_ratio = if g.mem_total_mib > 0 {
            g.mem_used_mib as f64 / g.mem_total_mib as f64
        } else { 0.0 };
        let color = if g.util_pct > 80 { Color::Red }
                    else if g.util_pct > 30 { Color::Yellow }
                    else { Color::Green };
        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(color).bg(Color::DarkGray))
            .ratio(mem_ratio.min(1.0).max(0.0))
            .label(format!("mem {:>3.0}%  util {}%", mem_ratio * 100.0, g.util_pct));
        f.render_widget(gauge, gauge_rect);
    }
}
