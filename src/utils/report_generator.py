# src/utils/report_generator.py
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive evaluation reports"""
    
    def __init__(self):
        self.report_template = self._get_report_template()
    
    async def generate_evaluation_report(self, evaluation_run, output_path: str):
        """Generate complete HTML evaluation report"""
        try:
            # Generate comprehensive report
            await self.generate_comprehensive_report(evaluation_run, output_path)
            logger.info(f"Report generated successfully: {output_path}")
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise
    
    async def generate_comprehensive_report(self, evaluation_run, output_path: str):
        """Generate comprehensive HTML report"""
        try:
            # Simple HTML report generation
            html_content = self._generate_simple_html_report(evaluation_run)
            
            # Save report
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Comprehensive report generated: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")
            raise
    
    def _generate_simple_html_report(self, evaluation_run) -> str:
        """Generate a simple HTML report"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>LLM Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ color: #333; border-bottom: 2px solid #eee; padding-bottom: 20px; }}
        .summary {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
        .model {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .score {{ font-weight: bold; color: #007bff; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>LLM Coding Evaluation Report</h1>
        <p>Run ID: {evaluation_run.run_id}</p>
        <p>Generated: {evaluation_run.timestamp}</p>
        <p>Duration: {evaluation_run.total_duration:.2f} seconds</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Models Evaluated: {len(evaluation_run.model_results)}</p>
        <p>Status: {evaluation_run.status}</p>
    </div>
    
    <div class="results">
        <h2>Model Results</h2>"""
        
        for model_result in evaluation_run.model_results:
            html += f"""
        <div class="model">
            <h3>{model_result.model_name}</h3>
            <p>Provider: {model_result.provider}</p>
            <p>Overall Score: <span class="score">{model_result.overall_score:.3f}</span></p>
            <p>Tasks Completed: {model_result.total_tasks}</p>
            <p>Tasks Passed: {model_result.passed_tasks}</p>
            <p>Execution Time: {model_result.execution_time:.2f}s</p>
        </div>"""
        
        html += """
    </div>
</body>
</html>"""
        
        return html
    
    def _generate_charts(self, evaluation_run) -> Dict[str, str]:
        """Generate all charts for the report"""
        charts = {}
        
        try:
            # Overall performance comparison
            charts['overall_performance'] = self._create_overall_performance_chart(evaluation_run)
            
            # Task type breakdown
            charts['task_type_breakdown'] = self._create_task_type_breakdown_chart(evaluation_run)
            
            # Difficulty analysis
            charts['difficulty_analysis'] = self._create_difficulty_analysis_chart(evaluation_run)
            
            # Execution time comparison
            charts['execution_time'] = self._create_execution_time_chart(evaluation_run)
            
            # Detailed metrics radar
            charts['metrics_radar'] = self._create_metrics_radar_chart(evaluation_run)
            
        except Exception as e:
            logger.error(f"Error generating charts: {e}")
            # Return empty charts dict if generation fails
            return {}
        
        return charts
    
    def _create_overall_performance_chart(self, evaluation_run) -> str:
        """Create overall performance comparison chart"""
        leaderboard = evaluation_run.get_leaderboard()
        
        if not leaderboard:
            return ""
        
        df = pd.DataFrame(leaderboard)
        
        fig = px.bar(
            df, 
            x='model_name', 
            y='overall_score',
            title='Overall Model Performance Comparison',
            color='overall_score',
            color_continuous_scale='Viridis',
            text='overall_score'
        )
        
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Overall Score",
            showlegend=False,
            height=400
        )
        
        return pyo.plot(fig, output_type='div', include_plotlyjs=False)
    
    def _create_task_type_breakdown_chart(self, evaluation_run) -> str:
        """Create task type performance breakdown"""
        leaderboard = evaluation_run.get_leaderboard()
        
        if not leaderboard:
            return ""
        
        df = pd.DataFrame(leaderboard)
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Frontend', 'Backend', 'Testing'),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # Frontend scores
        fig.add_trace(
            go.Bar(x=df['model_name'], y=df['frontend_score'], name='Frontend'),
            row=1, col=1
        )
        
        # Backend scores
        fig.add_trace(
            go.Bar(x=df['model_name'], y=df['backend_score'], name='Backend'),
            row=1, col=2
        )
        
        # Testing scores
        fig.add_trace(
            go.Bar(x=df['model_name'], y=df['testing_score'], name='Testing'),
            row=1, col=3
        )
        
        fig.update_layout(
            title_text="Performance by Task Type",
            showlegend=False,
            height=400
        )
        
        return pyo.plot(fig, output_type='div', include_plotlyjs=False)
    
    def _create_difficulty_analysis_chart(self, evaluation_run) -> str:
        """Create difficulty level analysis chart"""
        # Collect data by difficulty
        difficulty_data = {}
        
        for model_result in evaluation_run.model_results:
            model_name = model_result.model_name
            difficulty_data[model_name] = {'easy': 0, 'medium': 0, 'hard': 0}
            
            for task_result in model_result.task_results:
                difficulty = task_result.difficulty.value
                if difficulty in difficulty_data[model_name]:
                    difficulty_data[model_name][difficulty] = task_result.score
        
        if not difficulty_data:
            return ""
        
        df_data = []
        for model, scores in difficulty_data.items():
            for difficulty, score in scores.items():
                df_data.append({
                    'model': model,
                    'difficulty': difficulty,
                    'score': score
                })
        
        df = pd.DataFrame(df_data)
        
        fig = px.bar(
            df,
            x='model',
            y='score',
            color='difficulty',
            barmode='group',
            title='Performance by Difficulty Level'
        )
        
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Average Score",
            height=400
        )
        
        return pyo.plot(fig, output_type='div', include_plotlyjs=False)
    
    def _create_execution_time_chart(self, evaluation_run) -> str:
        """Create execution time comparison chart"""
        leaderboard = evaluation_run.get_leaderboard()
        
        if not leaderboard:
            return ""
        
        df = pd.DataFrame(leaderboard)
        
        fig = px.bar(
            df,
            x='model_name',
            y='avg_time',
            title='Average Execution Time by Model',
            color='avg_time',
            color_continuous_scale='Reds',
            text='avg_time'
        )
        
        fig.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
        fig.update_layout(
            xaxis_title="Model",
            yaxis_title="Average Time (seconds)",
            showlegend=False,
            height=400
        )
        
        return pyo.plot(fig, output_type='div', include_plotlyjs=False)
    
    def _create_metrics_radar_chart(self, evaluation_run) -> str:
        """Create detailed metrics radar chart"""
        # Get first model for demonstration (you'd loop through all models)
        if not evaluation_run.model_results:
            return ""
        
        model_result = evaluation_run.model_results[0]
        
        # Calculate average scores across metrics
        metrics = ['functionality', 'code_quality', 'security', 'performance', 'accessibility']
        scores = []
        
        for metric in metrics:
            total_score = 0
            count = 0
            for task_result in model_result.task_results:
                if task_result.detailed_scores and metric in task_result.detailed_scores:
                    total_score += task_result.detailed_scores[metric]
                    count += 1
            scores.append(total_score / max(count, 1))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=metrics,
            fill='toself',
            name=model_result.model_name
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Detailed Metrics Analysis",
            height=400
        )
        
        return pyo.plot(fig, output_type='div', include_plotlyjs=False)
    
    def _generate_summary_stats(self, evaluation_run) -> Dict[str, Any]:
        """Generate summary statistics"""
        stats = {
            'run_id': evaluation_run.run_id,
            'timestamp': evaluation_run.timestamp.isoformat(),
            'duration': f"{evaluation_run.duration:.2f} seconds",
            'total_models': len(evaluation_run.model_results),
            'total_tasks': sum(len(mr.task_results) for mr in evaluation_run.model_results),
            'leaderboard': evaluation_run.get_leaderboard()
        }
        
        # Calculate additional stats
        if evaluation_run.model_results:
            all_scores = []
            all_times = []
            
            for model_result in evaluation_run.model_results:
                for task_result in model_result.task_results:
                    all_scores.append(task_result.score)
                    all_times.append(task_result.execution_time)
            
            if all_scores:
                stats['avg_score'] = sum(all_scores) / len(all_scores)
                stats['avg_time'] = sum(all_times) / len(all_times)
                stats['success_rate'] = len([s for s in all_scores if s >= 0.7]) / len(all_scores)
        
        return stats
    
    def _generate_detailed_tables(self, evaluation_run) -> Dict[str, str]:
        """Generate detailed data tables"""
        tables = {}
        
        # Model comparison table
        leaderboard = evaluation_run.get_leaderboard()
        if leaderboard:
            df = pd.DataFrame(leaderboard)
            tables['model_comparison'] = df.to_html(
                classes='table table-striped table-hover',
                table_id='modelComparisonTable',
                escape=False,
                float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else x
            )
        
        # Detailed results table
        detailed_data = []
        for model_result in evaluation_run.model_results:
            for task_result in model_result.task_results:
                detailed_data.append({
                    'Model': task_result.model_name,
                    'Task': task_result.task_title,
                    'Type': task_result.task_type.value,
                    'Difficulty': task_result.difficulty.value,
                    'Score': f"{task_result.score:.3f}",
                    'Passed': '✅' if task_result.passed else '❌',
                    'Time (s)': f"{task_result.execution_time:.2f}",
                    'Error': task_result.error_message or ''
                })
        
        if detailed_data:
            df_detailed = pd.DataFrame(detailed_data)
            tables['detailed_results'] = df_detailed.to_html(
                classes='table table-striped table-hover table-sm',
                table_id='detailedResultsTable',
                escape=False
            )
        
        return tables
    
    def _generate_html_report(
        self, 
        evaluation_run, 
        charts: Dict[str, str], 
        summary_stats: Dict[str, Any], 
        tables: Dict[str, str]
    ) -> str:
        """Generate complete HTML report"""
        
        # Format timestamp
        timestamp = datetime.fromisoformat(summary_stats['timestamp'].replace('Z', '+00:00'))
        formatted_timestamp = timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Evaluation Report - {summary_stats['run_id'][:8]}</title>
    
    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    
    <style>
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            margin-bottom: 1rem;
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }}
        
        .metric-label {{
            font-size: 0.9rem;
            opacity: 0.9;
        }}
        
        .chart-container {{
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .leaderboard-item {{
            display: flex;
            align-items: center;
            padding: 1rem;
            margin-bottom: 0.5rem;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        
        .rank {{
            font-size: 1.5rem;
            font-weight: bold;
            margin-right: 1rem;
            min-width: 3rem;
        }}
        
        .rank-1 {{ color: #FFD700; }}
        .rank-2 {{ color: #C0C0C0; }}
        .rank-3 {{ color: #CD7F32; }}
        
        .table-container {{
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body style="background-color: #f8f9fa;">
    <div class="container-fluid py-4">
        <!-- Header -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="text-center">
                    <h1 class="display-4 fw-bold">LLM Coding Evaluation Report</h1>
                    <p class="lead text-muted">Run ID: {summary_stats['run_id']}</p>
                    <p class="text-muted">Generated on {formatted_timestamp}</p>
                </div>
            </div>
        </div>
        
        <!-- Summary Metrics -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value">{summary_stats['total_models']}</div>
                    <div class="metric-label">Models Evaluated</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value">{summary_stats['total_tasks']}</div>
                    <div class="metric-label">Total Tasks</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value">{summary_stats.get('avg_score', 0):.3f}</div>
                    <div class="metric-label">Average Score</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-value">{summary_stats['duration']}</div>
                    <div class="metric-label">Total Duration</div>
                </div>
            </div>
        </div>
        
        <!-- Leaderboard -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="chart-container">
                    <h3 class="mb-4">🏆 Leaderboard</h3>
"""
        
        # Add leaderboard items
        for i, entry in enumerate(summary_stats['leaderboard'][:5], 1):
            rank_class = f"rank-{i}" if i <= 3 else ""
            html_content += f"""
                    <div class="leaderboard-item">
                        <div class="rank {rank_class}">#{i}</div>
                        <div class="flex-grow-1">
                            <h5 class="mb-1">{entry['model_name']}</h5>
                            <small class="text-muted">{entry['provider']}</small>
                        </div>
                        <div class="text-end">
                            <div class="fw-bold">{entry['overall_score']:.3f}</div>
                            <small class="text-muted">{entry['pass_rate']:.1%} pass rate</small>
                        </div>
                    </div>
"""
        
        html_content += """
                </div>
            </div>
        </div>
        
        <!-- Charts -->
        <div class="row">
            <div class="col-12">
                <div class="chart-container">
                    <h3 class="mb-4">📊 Performance Analysis</h3>
"""
        
        # Add charts
        for chart_name, chart_html in charts.items():
            if chart_html:
                html_content += f'<div class="mb-4">{chart_html}</div>'
        
        html_content += """
                </div>
            </div>
        </div>
        
        <!-- Detailed Tables -->
        <div class="row">
            <div class="col-12">
                <div class="table-container">
                    <h3 class="mb-4">📋 Detailed Results</h3>
"""
        
        # Add tables
        for table_name, table_html in tables.items():
            if table_html:
                html_content += f'<div class="mb-4">{table_html}</div>'
        
        html_content += """
                </div>
            </div>
        </div>
        
        <!-- Footer -->
        <div class="row mt-5">
            <div class="col-12 text-center text-muted">
                <p>Generated by LLM Coding Evaluation Platform</p>
                <p><small>Report ID: {summary_stats['run_id']}</small></p>
            </div>
        </div>
    </div>
    
    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
        
        return html_content
    
    def _get_report_template(self) -> str:
        """Get base HTML template for reports"""
        # This could be loaded from a file in a real implementation
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLM Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { text-align: center; margin-bottom: 40px; }
                .metric { display: inline-block; margin: 20px; text-align: center; }
                .chart { margin: 20px 0; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            {content}
        </body>
        </html>
        """


# Example usage
if __name__ == "__main__":
    # This would be used with actual evaluation run data
    generator = ReportGenerator()
    print("Report generator initialized successfully!")