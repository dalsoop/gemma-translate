use crate::app::App;
use ratatui::{
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, List, ListItem},
    Frame,
};

pub fn draw(f: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .title(" 로그 ")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded);
    let inner = block.inner(area);
    f.render_widget(block, area);

    let take = inner.height as usize;
    let items: Vec<ListItem> = app
        .log
        .iter()
        .rev()
        .take(take)
        .rev()
        .map(|line| {
            let color = if line.contains("FAILED") || line.contains("error") {
                Color::Red
            } else if line.contains("done") {
                Color::Green
            } else {
                Color::Gray
            };
            ListItem::new(Line::from(Span::styled(line.clone(), Style::default().fg(color))))
        })
        .collect();
    f.render_widget(List::new(items), inner);
}
