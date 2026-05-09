"""
analysis_dashboard.py — All 7 visualizations from emotion_log.csv
Generates base64-encoded PNG images for the Flask dashboard.

Visualizations:
1. V-A Scatter Plot (colored by zone)
2. Stress Score Timeline
3. Rolling Average Stress Level (20s window)
4. Zone Distribution Pie Chart
5. Daily Average Stress (bar chart)
6. Valence & Arousal Timelines (stacked)
7. Acoustic Feature Trends (Pitch, Jitter, Shimmer)
"""

import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────────────────
#  Zone Colors
# ──────────────────────────────────────────────────────────────
ZONE_COLORS = {
    'high_stress': '#ef4444',  # Red
    'anxiety':     '#f97316',  # Orange
    'anger':       '#dc2626',  # Dark Red
    'sadness':     '#6366f1',  # Indigo
    'joy':         '#22c55e',  # Green
    'calm':        '#3b82f6',  # Blue
    'neutral':     '#9ca3af',  # Gray
}

ZONE_LABELS = {
    'high_stress': 'High Stress',
    'anxiety':     'Anxiety',
    'anger':       'Anger',
    'sadness':     'Sadness',
    'joy':         'Joy',
    'calm':        'Calm',
    'neutral':     'Neutral',
}


def fig_to_base64(fig):
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str


def set_dark_style():
    """Set matplotlib style for dark dashboard."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#1a1a2e',
        'axes.facecolor': '#16213e',
        'axes.edgecolor': '#e2e8f0',
        'axes.labelcolor': '#e2e8f0',
        'text.color': '#e2e8f0',
        'xtick.color': '#e2e8f0',
        'ytick.color': '#e2e8f0',
        'grid.color': '#334155',
        'grid.alpha': 0.3,
        'font.family': 'sans-serif',
        'font.size': 11,
    })


def load_emotion_log(filepath='emotion_log.csv'):
    """Load and preprocess CSV log."""
    if filepath is None or not os.path.exists(filepath):
        return generate_demo_data()

    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def generate_demo_data(n=500):
    """Generate demo data if no CSV exists."""
    np.random.seed(42)
    now = datetime.now()
    timestamps = [now - timedelta(seconds=i*2) for i in range(n)]
    timestamps.reverse()

    valence = np.cumsum(np.random.randn(n) * 0.05)
    valence = np.clip(valence / (np.abs(valence).max() + 1e-6), -1, 1)
    arousal = np.cumsum(np.random.randn(n) * 0.05)
    arousal = np.clip(arousal / (np.abs(arousal).max() + 1e-6), -1, 1)

    stress_scores = np.clip((-valence * 0.5 + arousal * 0.5 + 0.3), 0, 1)

    zones = []
    for v, a in zip(valence, arousal):
        if a > 0.3 and v < -0.2:
            zones.append('high_stress')
        elif a > 0.5 and v < 0.0:
            zones.append('anxiety')
        elif a > 0.4 and v < -0.3:
            zones.append('anger')
        elif a < -0.1 and v < -0.2:
            zones.append('sadness')
        elif a > 0.3 and v > 0.3:
            zones.append('joy')
        elif a < 0.1 and v > 0.1:
            zones.append('calm')
        else:
            zones.append('neutral')

    # Acoustic features
    pitch   = 150 + np.cumsum(np.random.randn(n) * 2)
    jitter  = np.abs(0.02 + np.cumsum(np.random.randn(n) * 0.001))
    shimmer = np.abs(0.05 + np.cumsum(np.random.randn(n) * 0.002))

    df = pd.DataFrame({
        'timestamp': timestamps,
        'valence': valence,
        'arousal': arousal,
        'stress_score': stress_scores,
        'dominant_zone': zones,
        'pitch': pitch,
        'jitter': jitter,
        'shimmer': shimmer,
    })
    return df


# ──────────────────────────────────────────────────────────────
#  Plot 1: V-A Scatter Plot
# ──────────────────────────────────────────────────────────────
def plot_va_scatter(df, neutral_v=0.1, neutral_a=-0.1):
    """V-A scatter plot colored by zone with neutral baseline marked."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    for zone, color in ZONE_COLORS.items():
        mask = df['dominant_zone'] == zone
        if mask.sum() > 0:
            ax.scatter(df.loc[mask, 'valence'], df.loc[mask, 'arousal'],
                      c=color, label=ZONE_LABELS.get(zone, zone),
                      alpha=0.6, s=30, edgecolors='white', linewidth=0.3)

    # Mark neutral baseline
    ax.scatter(neutral_v, neutral_a, marker='X', c='#fbbf24', s=200,
              zorder=10, edgecolors='white', linewidth=2, label='Neutral Baseline')

    # Quadrant lines
    ax.axhline(y=0, color='#64748b', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='#64748b', linestyle='--', alpha=0.5)

    # Zone labels
    ax.text(-0.8, 0.8, 'STRESS', fontsize=12, color='#ef4444', fontweight='bold', alpha=0.7)
    ax.text(0.5, 0.8, 'JOY', fontsize=12, color='#22c55e', fontweight='bold', alpha=0.7)
    ax.text(-0.8, -0.8, 'SAD', fontsize=12, color='#6366f1', fontweight='bold', alpha=0.7)
    ax.text(0.5, -0.8, 'CALM', fontsize=12, color='#3b82f6', fontweight='bold', alpha=0.7)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('Valence →', fontsize=13, fontweight='bold')
    ax.set_ylabel('Arousal →', fontsize=13, fontweight='bold')
    ax.set_title('Valence-Arousal Space', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.2)

    return fig_to_base64(fig)


# ──────────────────────────────────────────────────────────────
#  Plot 2: Stress Score Timeline
# ──────────────────────────────────────────────────────────────
def plot_stress_timeline(df):
    """Stress score vs time with threshold line."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(df['timestamp'], df['stress_score'], alpha=0.3, color='#f43f5e')
    ax.plot(df['timestamp'], df['stress_score'], color='#f43f5e', linewidth=1.5, label='Stress Score')
    ax.axhline(y=0.5, color='#fbbf24', linestyle='--', linewidth=2, label='Threshold (0.5)', alpha=0.8)

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Stress Score', fontsize=12, fontweight='bold')
    ax.set_title('Stress Score Timeline', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    fig.autofmt_xdate()
    return fig_to_base64(fig)


# ──────────────────────────────────────────────────────────────
#  Plot 3: Rolling Average Stress
# ──────────────────────────────────────────────────────────────
def plot_rolling_stress(df, window=10):
    """Rolling average stress with 20-second sliding window."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    rolling = df['stress_score'].rolling(window=window, min_periods=1).mean()

    ax.plot(df['timestamp'], df['stress_score'], color='#f43f5e', alpha=0.3, linewidth=0.8, label='Raw')
    ax.plot(df['timestamp'], rolling, color='#8b5cf6', linewidth=2.5, label=f'Rolling Avg (w={window})')
    ax.axhline(y=0.5, color='#fbbf24', linestyle='--', linewidth=1.5, alpha=0.7)

    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Stress Score', fontsize=12, fontweight='bold')
    ax.set_title('Rolling Average Stress Level', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    fig.autofmt_xdate()
    return fig_to_base64(fig)


# ──────────────────────────────────────────────────────────────
#  Plot 4: Zone Distribution Pie Chart
# ──────────────────────────────────────────────────────────────
def plot_zone_distribution(df):
    """Pie chart of time spent in each V-A zone (legend-based to avoid overlap)."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(9, 7))

    zone_counts = df['dominant_zone'].value_counts()
    colors = [ZONE_COLORS.get(z, '#9ca3af') for z in zone_counts.index]
    labels = [ZONE_LABELS.get(z, z) for z in zone_counts.index]

    wedges, _, autotexts = ax.pie(
        zone_counts.values,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.78,
        wedgeprops=dict(width=0.5, edgecolor='#1a1a2e', linewidth=2),
    )

    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight('bold')
        autotext.set_color('#e2e8f0')

    # Use a legend instead of labels on wedges to prevent overlap
    ax.legend(
        wedges, labels,
        title='Zones',
        loc='center left',
        bbox_to_anchor=(0.85, 0, 0.5, 1),
        fontsize=10,
        title_fontsize=11,
        framealpha=0.3,
        edgecolor='#334155',
    )

    ax.set_title('Zone Distribution', fontsize=16, fontweight='bold', pad=15)
    plt.tight_layout()

    return fig_to_base64(fig)


# ──────────────────────────────────────────────────────────────
#  Plot 5: Daily Average Stress
# ──────────────────────────────────────────────────────────────
def plot_daily_stress(df):
    """Bar chart of daily average stress scores."""
    set_dark_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    df_copy = df.copy()
    df_copy['date'] = df_copy['timestamp'].dt.date
    daily = df_copy.groupby('date')['stress_score'].mean()

    colors = ['#ef4444' if s > 0.5 else '#22c55e' for s in daily.values]
    bars = ax.bar(range(len(daily)), daily.values, color=colors,
                  edgecolor='white', linewidth=0.5, alpha=0.85)

    ax.set_xticks(range(len(daily)))
    ax.set_xticklabels([str(d) for d in daily.index], rotation=45, ha='right', fontsize=9)
    ax.axhline(y=0.5, color='#fbbf24', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold')

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Avg Stress Score', fontsize=12, fontweight='bold')
    ax.set_title('Daily Average Stress', fontsize=16, fontweight='bold', pad=15)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    return fig_to_base64(fig)


# ──────────────────────────────────────────────────────────────
#  Plot 6: Valence & Arousal Timelines (stacked)
# ──────────────────────────────────────────────────────────────
def plot_va_timelines(df):
    """Two stacked line charts for valence and arousal over session."""
    set_dark_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Valence
    ax1.fill_between(df['timestamp'], df['valence'], alpha=0.3, color='#6366f1')
    ax1.plot(df['timestamp'], df['valence'], color='#6366f1', linewidth=1.5)
    ax1.axhline(y=0, color='#64748b', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Valence', fontsize=12, fontweight='bold')
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_title('Valence & Arousal Timelines', fontsize=16, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.2)
    ax1.text(0.02, 0.95, 'VALENCE', transform=ax1.transAxes, fontsize=14,
            fontweight='bold', color='#6366f1', va='top')

    # Arousal
    ax2.fill_between(df['timestamp'], df['arousal'], alpha=0.3, color='#f43f5e')
    ax2.plot(df['timestamp'], df['arousal'], color='#f43f5e', linewidth=1.5)
    ax2.axhline(y=0, color='#64748b', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Arousal', fontsize=12, fontweight='bold')
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.2)
    ax2.text(0.02, 0.95, 'AROUSAL', transform=ax2.transAxes, fontsize=14,
            fontweight='bold', color='#f43f5e', va='top')

    fig.autofmt_xdate()
    plt.tight_layout()
    return fig_to_base64(fig)


# ──────────────────────────────────────────────────────────────
#  Plot 7: Acoustic Feature Trends
# ──────────────────────────────────────────────────────────────
def plot_acoustic_trends(df):
    """Three sub-plots: Pitch, Jitter, Shimmer over session time."""
    set_dark_style()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Check if acoustic columns exist
    has_pitch = 'pitch' in df.columns
    has_jitter = 'jitter' in df.columns
    has_shimmer = 'shimmer' in df.columns

    # Pitch
    if has_pitch:
        ax1.plot(df['timestamp'], df['pitch'], color='#22d3ee', linewidth=1.5)
        ax1.fill_between(df['timestamp'], df['pitch'], alpha=0.2, color='#22d3ee')
    ax1.set_ylabel('Pitch (Hz)', fontsize=11, fontweight='bold')
    ax1.set_title('Acoustic Feature Trends', fontsize=16, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.2)
    ax1.text(0.02, 0.95, 'PITCH', transform=ax1.transAxes, fontsize=13,
            fontweight='bold', color='#22d3ee', va='top')

    # Jitter
    if has_jitter:
        ax2.plot(df['timestamp'], df['jitter'], color='#fb923c', linewidth=1.5)
        ax2.fill_between(df['timestamp'], df['jitter'], alpha=0.2, color='#fb923c')
    ax2.set_ylabel('Jitter', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.2)
    ax2.text(0.02, 0.95, 'JITTER', transform=ax2.transAxes, fontsize=13,
            fontweight='bold', color='#fb923c', va='top')

    # Shimmer
    if has_shimmer:
        ax3.plot(df['timestamp'], df['shimmer'], color='#a78bfa', linewidth=1.5)
        ax3.fill_between(df['timestamp'], df['shimmer'], alpha=0.2, color='#a78bfa')
    ax3.set_xlabel('Time', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Shimmer', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.2)
    ax3.text(0.02, 0.95, 'SHIMMER', transform=ax3.transAxes, fontsize=13,
            fontweight='bold', color='#a78bfa', va='top')

    fig.autofmt_xdate()
    plt.tight_layout()
    return fig_to_base64(fig)


# ──────────────────────────────────────────────────────────────
#  Mood Report Generator
# ──────────────────────────────────────────────────────────────
def generate_mood_report(df):
    """
    Generate a comprehensive mood report with per-zone breakdown,
    overall assessment, and personalized recommendations.

    Returns:
        dict with zone_breakdown, overall_mood, and recommendations
    """
    total = len(df)
    if total == 0:
        return {
            'zone_breakdown': [],
            'overall_mood': 'No data available yet.',
            'mood_emoji': '📊',
            'recommendations': ['Start a live analysis session to generate your mood report.'],
            'past_sessions': [],
        }

    # Per-zone breakdown
    zone_counts = df['dominant_zone'].value_counts()
    zone_breakdown = []
    for zone_key, count in zone_counts.items():
        pct = (count / total) * 100
        zone_breakdown.append({
            'zone': zone_key,
            'label': ZONE_LABELS.get(zone_key, zone_key),
            'color': ZONE_COLORS.get(zone_key, '#9ca3af'),
            'count': int(count),
            'percentage': round(pct, 1),
        })

    # Overall mood assessment
    dominant = zone_counts.index[0]
    avg_stress = float(df['stress_score'].mean())
    avg_valence = float(df['valence'].mean())
    stress_pct = float((df['dominant_zone'] == 'high_stress').mean() * 100)
    anxiety_pct = float((df['dominant_zone'] == 'anxiety').mean() * 100)
    joy_pct = float((df['dominant_zone'] == 'joy').mean() * 100)
    calm_pct = float((df['dominant_zone'] == 'calm').mean() * 100)
    negative_pct = stress_pct + anxiety_pct + float((df['dominant_zone'] == 'anger').mean() * 100) + float((df['dominant_zone'] == 'sadness').mean() * 100)

    if avg_stress > 0.7:
        overall_mood = 'Your stress levels have been significantly elevated. Immediate stress-management steps are strongly recommended.'
        mood_emoji = '🔴'
    elif avg_stress > 0.5:
        overall_mood = 'You are experiencing moderate stress. Consider taking breaks and practicing relaxation techniques.'
        mood_emoji = '🟠'
    elif negative_pct > 50:
        overall_mood = 'Your session shows a mix of negative emotional states. Mindfulness and self-care practices can help restore balance.'
        mood_emoji = '🟡'
    elif joy_pct + calm_pct > 60:
        overall_mood = 'You are in a very positive emotional state! Keep up whatever you are doing — your wellbeing indicators are excellent.'
        mood_emoji = '🟢'
    elif avg_valence > 0.1:
        overall_mood = 'Your emotional state is generally positive with balanced arousal. You are in a healthy, productive zone.'
        mood_emoji = '🟢'
    else:
        overall_mood = 'Your emotional state is relatively neutral. Regular monitoring will help you track trends over time.'
        mood_emoji = '🔵'

    # Personalized recommendations based on dominant moods
    recommendations = []
    if stress_pct > 20:
        recommendations.extend([
            '🧘 Practice deep breathing: inhale 4s, hold 4s, exhale 6s — repeat 5 times',
            '🚶 Take a 10-minute walk outdoors to lower cortisol levels',
            '📵 Schedule digital detox periods — 30 min away from screens',
        ])
    if anxiety_pct > 15:
        recommendations.extend([
            '✍️ Try journaling: write down 3 worries and possible solutions for each',
            '🎵 Listen to calming music (60-80 BPM) for 15 minutes',
            '🌿 Practice the 5-4-3-2-1 grounding technique when feeling anxious',
        ])
    if float((df['dominant_zone'] == 'anger').mean() * 100) > 10:
        recommendations.extend([
            '🏋️ Channel energy into physical exercise — even 5 minutes helps',
            '🧊 Apply cold water to your face or wrists to activate the calming dive reflex',
        ])
    if float((df['dominant_zone'] == 'sadness').mean() * 100) > 15:
        recommendations.extend([
            '👥 Reach out to a friend or loved one — social connection helps resilience',
            '🌅 Spend time in natural sunlight to boost serotonin levels',
            '📋 Write down 3 things you are grateful for today',
        ])
    if joy_pct > 30:
        recommendations.extend([
            '🎯 Use this positive momentum for creative or challenging tasks',
            '📝 Note what activities led to this positive state — replicate them',
        ])
    if calm_pct > 30:
        recommendations.extend([
            '📚 This calm state is ideal for deep, focused work — capitalize on it',
            '🧠 Practice mindfulness to sustain and extend this balanced state',
        ])
    if not recommendations:
        recommendations = [
            '💧 Stay hydrated — drink water regularly throughout the day',
            '⏰ Take a 5-minute break every 30 minutes to maintain balance',
            '🎯 Set small, achievable goals to build positive momentum',
            '😴 Ensure 7-8 hours of quality sleep for emotional regulation',
        ]

    # Past sessions summary (group by date)
    df_copy = df.copy()
    df_copy['date'] = df_copy['timestamp'].dt.date
    past_sessions = []
    for date, group in df_copy.groupby('date'):
        session_dominant = group['dominant_zone'].mode().iloc[0] if len(group) > 0 else 'neutral'
        past_sessions.append({
            'date': str(date),
            'samples': len(group),
            'avg_stress': round(float(group['stress_score'].mean()), 2),
            'max_stress': round(float(group['stress_score'].max()), 2),
            'avg_valence': round(float(group['valence'].mean()), 2),
            'avg_arousal': round(float(group['arousal'].mean()), 2),
            'dominant_zone': session_dominant,
            'dominant_label': ZONE_LABELS.get(session_dominant, session_dominant),
            'dominant_color': ZONE_COLORS.get(session_dominant, '#9ca3af'),
            'duration': str(group['timestamp'].max() - group['timestamp'].min()) if len(group) > 1 else '< 1 min',
        })
    past_sessions.reverse()  # Most recent first

    return {
        'zone_breakdown': zone_breakdown,
        'overall_mood': overall_mood,
        'mood_emoji': mood_emoji,
        'recommendations': recommendations,
        'past_sessions': past_sessions,
    }


# ──────────────────────────────────────────────────────────────
#  Generate All Dashboard Plots
# ──────────────────────────────────────────────────────────────
def generate_all_plots(csv_path='emotion_log.csv', neutral_v=0.1, neutral_a=-0.1):
    """
    Generate all 7 dashboard visualizations + mood report.

    Returns:
        dict of plot_name -> base64 PNG string
        dict of stats
    """
    df = load_emotion_log(csv_path)

    plots = {
        'va_scatter':       plot_va_scatter(df, neutral_v, neutral_a),
        'stress_timeline':  plot_stress_timeline(df),
        'rolling_stress':   plot_rolling_stress(df),
        'zone_distribution': plot_zone_distribution(df),
        'daily_stress':     plot_daily_stress(df),
        'va_timelines':     plot_va_timelines(df),
        'acoustic_trends':  plot_acoustic_trends(df),
    }

    # Compute summary stats
    stats = {
        'avg_stress': float(df['stress_score'].mean()),
        'max_stress': float(df['stress_score'].max()),
        'min_stress': float(df['stress_score'].min()),
        'avg_valence': float(df['valence'].mean()),
        'avg_arousal': float(df['arousal'].mean()),
        'total_samples': len(df),
        'dominant_zone': df['dominant_zone'].mode().iloc[0] if len(df) > 0 else 'neutral',
        'high_stress_pct': float((df['dominant_zone'] == 'high_stress').mean() * 100),
        'session_duration': str(df['timestamp'].max() - df['timestamp'].min()) if len(df) > 1 else '0:00:00',
    }

    # Generate mood report
    stats['mood_report'] = generate_mood_report(df)

    return plots, stats


# ──────────────────────────────────────────────────────────────
#  Demo
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Analytics Dashboard — Demo")
    print("=" * 60)

    plots, stats = generate_all_plots()

    print(f"\n📊 Generated {len(plots)} plots:")
    for name, b64 in plots.items():
        print(f"  • {name}: {len(b64)} bytes (base64)")

    print(f"\n📈 Session Stats:")
    for key, val in stats.items():
        print(f"  • {key}: {val}")

    print("\n✅ Dashboard generation complete!")
