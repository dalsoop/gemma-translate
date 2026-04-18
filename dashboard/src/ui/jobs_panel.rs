use crate::app::App;
use crate::jobs::JobStatus;
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Gauge, List, ListItem, Paragraph},
    Frame,
};

pub fn draw_active(f: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .title(format!(" 실행/대기 ({}) ", app.active.len() + app.queue.len()))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let mut rows: Vec<(String, JobStatus, f32, bool)> = Vec::new();
    for (i, j) in app.active.iter().enumerate() {
        rows.push((j.kind.title(), j.status, j.progress, i == app.selected_active));
    }
    for j in &app.queue {
        rows.push((j.kind.title(), j.status, j.progress, false));
    }

    if rows.is_empty() {
        f.render_widget(
            Paragraph::new("작업 없음. [n] 으로 추가").style(Style::default().fg(Color::DarkGray)),
            inner,
        );
        return;
    }

    for (i, (title, st, prog, selected)) in rows.iter().enumerate() {
        let y = inner.y + i as u16 * 2;
        if y + 1 >= inner.y + inner.height { break; }
        let label_rect = Rect { x: inner.x, y, width: inner.width, height: 1 };
        let gauge_rect = Rect { x: inner.x, y: y + 1, width: inner.width, height: 1 };

        let color = match st {
            JobStatus::Running => Color::Green,
            JobStatus::Queued => Color::DarkGray,
            _ => Color::White,
        };
        let mut label_style = Style::default().fg(color).add_modifier(Modifier::BOLD);
        let mut title_style = Style::default();
        if *selected {
            label_style = label_style.bg(Color::DarkGray);
            title_style = title_style.bg(Color::DarkGray).add_modifier(Modifier::BOLD);
        }
        let line = Line::from(vec![
            Span::styled(format!(" {} ", st.symbol()), label_style),
            Span::styled(title.clone(), title_style),
        ]);
        f.render_widget(Paragraph::new(line), label_rect);

        let gauge = Gauge::default()
            .gauge_style(Style::default().fg(color).bg(Color::DarkGray))
            .ratio((*prog as f64).min(1.0).max(0.0))
            .label(format!("{:.0}%", prog * 100.0));
        f.render_widget(gauge, gauge_rect);
    }
}

pub fn draw_history(f: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .title(format!(" 히스토리 ({}) ", app.history.len()))
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let items: Vec<ListItem> = app.history.iter().take(inner.height as usize).map(|j| {
        let st_color = match j.status {
            JobStatus::Done => Color::Green,
            JobStatus::Failed => Color::Red,
            JobStatus::Cancelled => Color::Yellow,
            _ => Color::DarkGray,
        };
        let duration = match (j.started_at, j.finished_at) {
            (Some(s), Some(e)) => format!("{:>4}s", (e - s).num_seconds()),
            _ => "   -".into(),
        };
        ListItem::new(Line::from(vec![
            Span::styled(format!(" {} ", j.status.symbol()), Style::default().fg(st_color)),
            Span::raw(format!("{duration}  ")),
            Span::raw(j.kind.title()),
        ]))
    }).collect();
    f.render_widget(List::new(items), inner);
}
