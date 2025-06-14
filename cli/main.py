"""Main CLI entry point using Typer."""

import typer
import asyncio
from rich.console import Console
from typing import Optional
from cli.chat import run_interactive, quick_chat
from cli.client import ApiClient

app = typer.Typer(
    help="Agentic Ecommerce Chatbot CLI",
    add_completion=False,
    rich_markup_mode="rich"
)
console = Console()


@app.command()
def chat():
    """
    Start an interactive chat session with the shopping assistant.
    
    [bold cyan]Example:[/bold cyan]
        python -m cli chat
    """
    console.print("[bold cyan]Starting interactive chat...[/bold cyan]\n")
    asyncio.run(run_interactive())


@app.command()
def quick(message: str):
    """
    Send a quick message without entering interactive mode.
    
    [bold cyan]Example:[/bold cyan]
        python -m cli quick "I need business casual shirts"
    """
    asyncio.run(quick_chat(message))


@app.command()
def products(
    category: Optional[str] = typer.Option(None, "--category", "-c", help="Filter by category"),
    style: Optional[str] = typer.Option(None, "--style", "-s", help="Filter by style"),
    max_price: Optional[float] = typer.Option(None, "--max-price", "-p", help="Maximum price")
):
    """
    Browse available products with optional filters.
    
    [bold cyan]Examples:[/bold cyan]
        python -m cli products
        python -m cli products --category shirts --style casual
        python -m cli products --max-price 100
    """
    async def show_products():
        async with ApiClient() as client:
            try:
                filters = {}
                if category:
                    filters["category"] = category
                if style:
                    filters["style"] = style
                if max_price:
                    filters["max_price"] = max_price
                
                with console.status("[bold green]Loading products..."):
                    response = await client.get_products(filters)
                
                products = response["products"]
                
                # Display in a nice table
                from rich.table import Table
                from rich import box
                
                table = Table(
                    title="Products",
                    box=box.ROUNDED,
                    show_header=True,
                    header_style="bold cyan"
                )
                
                table.add_column("Name", style="white")
                table.add_column("Category", style="magenta")
                table.add_column("Style", style="cyan")
                table.add_column("Price", justify="right", style="green")
                table.add_column("Rating", justify="center", style="yellow")
                
                for product in products:
                    rating_display = f"{product['rating']:.1f}/5"
                    table.add_row(
                        product["name"],
                        product["category"],
                        product["style"],
                        f"${product['price']:.2f}",
                        rating_display
                    )
                
                console.print(table)
                console.print(f"\n[dim]Total: {response['total']} products[/dim]")
                
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    asyncio.run(show_products())


@app.command()
def stats():
    """
    Show recommendation graph statistics.
    
    [bold cyan]Example:[/bold cyan]
        python -m cli stats
    """
    async def show_stats():
        async with ApiClient() as client:
            try:
                with console.status("[bold green]Loading statistics..."):
                    stats = await client.get_graph_stats()
                
                from rich.panel import Panel
                
                stats_text = f"""[bold]Graph Overview:[/bold]
                
Total Nodes: [cyan]{stats['total_nodes']}[/cyan]
Total Edges: [cyan]{stats['total_edges']}[/cyan]
Active Sessions: [green]{stats['active_sessions']}[/green]

[bold]Node Distribution:[/bold]
""" + "\n".join(f"  - {k}: {v}" for k, v in stats['node_counts'].items()) + """

[bold]Edge Distribution:[/bold]
""" + "\n".join(f"  - {k}: {v}" for k, v in stats['edge_counts'].items())
                
                panel = Panel(
                    stats_text,
                    title="Graph Statistics",
                    border_style="cyan"
                )
                console.print(panel)
                
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    asyncio.run(show_stats())


@app.command()
def demo():
    """
    Run a demo conversation for YouTube recording.
    
    Shows a scripted shopping experience with the AI assistant.
    """
    console.print("[bold cyan]Demo Mode - Agentic Ecommerce Chatbot[/bold cyan]\n")
    console.print("[dim]This will show a typical shopping conversation...[/dim]\n")
    
    async def run_demo():
        async with ApiClient() as client:
            try:
                # Check service
                await client.health_check()
                
                # Demo messages
                demo_messages = [
                    "Hi! I'm looking for some new clothes for work",
                    "I prefer business casual style, something modern but professional",
                    "I like navy and white colors, they go well with my wardrobe",
                    "My budget is around $80 per item",
                    "Show me what you recommend!"
                ]
                
                session_id = None
                
                for message in demo_messages:
                    # Show user message
                    console.print(f"[bold blue]You:[/bold blue] {message}")
                    
                    # Small pause for effect
                    await asyncio.sleep(1)
                    
                    # Get response
                    with console.status("[bold green]AI thinking..."):
                        response = await client.send_message(message, session_id)
                        session_id = response["session_id"]
                    
                    # Show response
                    console.print(f"\n[bold green]Assistant:[/bold green] {response['message']}")
                    
                    # Show recommendations if any
                    if response.get("recommendations"):
                        from cli.chat import ChatInterface
                        interface = ChatInterface(client)
                        interface.display_recommendations(response["recommendations"])
                    
                    console.print("\n" + "="*60 + "\n")
                    await asyncio.sleep(2)
                
                # Show final stats
                console.print("[bold cyan]Session complete! Let's look at the graph updates...[/bold cyan]\n")
                stats = await client.get_graph_stats()
                console.print(f"[green]>[/green] Graph now has {stats['total_edges']} edges")
                console.print(f"[green]>[/green] User preferences were learned and stored")
                console.print(f"[green]>[/green] Recommendations improved with each message\n")
                
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    asyncio.run(run_demo())


@app.callback()
def main():
    """
    [bold]Agentic Ecommerce Chatbot CLI[/bold]
    
    An AI-powered shopping assistant using LangGraph and Graph Neural Networks.
    
    Make sure the API service is running first:
        [cyan]python -m app.main[/cyan]
    
    Then use the CLI:
        [cyan]python -m cli chat[/cyan]
    """
    pass


if __name__ == "__main__":
    app()