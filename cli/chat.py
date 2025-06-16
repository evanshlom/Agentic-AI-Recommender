"""Interactive chat interface for the CLI."""

import asyncio
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.columns import Columns
from rich import box
from typing import Dict, Any, List
from cli.client import ApiClient
import httpx


console = Console()


class ChatInterface:
    """Rich-based chat interface."""
    
    def __init__(self, client: ApiClient):
        self.client = client
        self.session_id = None
        self.messages = []
    
    def display_welcome(self):
        """Display welcome message."""
        welcome = Panel(
            "[bold cyan]Welcome to the Agentic Ecommerce Chatbot![/bold cyan]\n\n"
            "[white]I'm your AI shopping assistant powered by Graph Neural Networks.[/white]\n"
            "[dim]Type 'help' for commands, 'exit' to quit.[/dim]",
            title="Shopping Assistant",
            border_style="cyan"
        )
        console.print(welcome)
    
    def display_message(self, role: str, content: str):
        """Display a chat message."""
        if role == "user":
            console.print(f"\n[bold blue]You:[/bold blue] {content}")
        else:
            console.print(f"\n[bold green]Assistant:[/bold green] {content}")
    
    def display_recommendations(self, recommendations: List[Dict[str, Any]]):
        """Display product recommendations in a table."""
        if not recommendations:
            return
        
        table = Table(
            title="Recommended Products",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
            title_style="bold cyan"
        )
        
        table.add_column("Product", style="white", no_wrap=True)
        table.add_column("Style", style="cyan")
        table.add_column("Price", justify="right", style="green")
        table.add_column("Rating", justify="center", style="yellow")
        table.add_column("Colors", style="blue")
        table.add_column("Match", justify="center", style="magenta")
        
        for rec in recommendations[:5]:  # Show top 5
            rating_display = f"{rec['rating']:.1f}/5"
            match_pct = f"{int(rec['score'] * 100)}%"
            colors = ", ".join(rec["colors"][:3])
            
            table.add_row(
                rec["name"],
                rec["style"],
                f"${rec['price']:.2f}",
                rating_display,
                colors,
                match_pct
            )
        
        console.print("\n")
        console.print(table)
        
        # Show reasons
        console.print("\n[bold cyan]Why these recommendations:[/bold cyan]")
        for i, rec in enumerate(recommendations[:3], 1):
            console.print(f"  {i}. [white]{rec['name']}[/white]: [dim]{rec['reason']}[/dim]")
    
    def display_help(self):
        """Display help information."""
        help_panel = Panel(
            "[bold]Available Commands:[/bold]\n\n"
            "  [cyan]help[/cyan]     - Show this help message\n"
            "  [cyan]products[/cyan] - Browse all products\n"
            "  [cyan]stats[/cyan]    - Show recommendation graph statistics\n"
            "  [cyan]clear[/cyan]    - Clear chat history\n"
            "  [cyan]exit[/cyan]     - Exit the chat\n\n"
            "[bold]Shopping Tips:[/bold]\n"
            "  - Be specific about styles, colors, and occasions\n"
            "  - Mention your budget if you have one\n"
            "  - Ask for similar items if you like something",
            title="Help",
            border_style="yellow"
        )
        console.print(help_panel)
    
    async def display_products(self):
        """Display all available products."""
        with console.status("[bold green]Loading products..."):
            response = await self.client.get_products()
        
        products = response["products"]
        
        table = Table(
            title="Available Products",
            box=box.DOUBLE_EDGE,
            show_header=True,
            header_style="bold cyan"
        )
        
        table.add_column("Category", style="magenta")
        table.add_column("Product", style="white")
        table.add_column("Style", style="cyan")
        table.add_column("Price", justify="right", style="green")
        table.add_column("Rating", justify="center", style="yellow")
        table.add_column("Brand", style="blue")
        
        for product in products:
            rating_display = f"{product['rating']:.1f}/5"
            
            table.add_row(
                product["category"],
                product["name"],
                product["style"],
                f"${product['price']:.2f}",
                rating_display,
                product["brand"]
            )
        
        console.print("\n")
        console.print(table)
        console.print(f"\n[dim]Total products: {response['total']}[/dim]")
    
    async def display_stats(self):
        """Display graph statistics."""
        with console.status("[bold green]Loading graph statistics..."):
            stats = await self.client.get_graph_stats()
        
        stats_panel = Panel(
            f"[bold]Graph Statistics:[/bold]\n\n"
            f"Total Nodes: [cyan]{stats['total_nodes']}[/cyan]\n"
            f"Total Edges: [cyan]{stats['total_edges']}[/cyan]\n\n"
            f"[bold]Node Types:[/bold]\n" +
            "\n".join(f"  - {k}: {v}" for k, v in stats['node_counts'].items()) +
            f"\n\n[bold]Edge Types:[/bold]\n" +
            "\n".join(f"  - {k}: {v}" for k, v in stats['edge_counts'].items()) +
            f"\n\n[bold]Active Sessions:[/bold] [green]{stats['active_sessions']}[/green]",
            title="Recommendation Graph",
            border_style="cyan"
        )
        console.print(stats_panel)
    
    async def run(self):
        """Run the interactive chat interface."""
        self.display_welcome()
        
        while True:
            try:
                # Get user input
                user_input = Prompt.ask("\n[bold blue]You[/bold blue]")
                
                # Handle commands
                if user_input.lower() == "exit":
                    console.print("\n[bold cyan]Thanks for shopping with us! Goodbye![/bold cyan]")
                    break
                elif user_input.lower() == "help":
                    self.display_help()
                    continue
                elif user_input.lower() == "products":
                    await self.display_products()
                    continue
                elif user_input.lower() == "stats":
                    await self.display_stats()
                    continue
                elif user_input.lower() == "clear":
                    console.clear()
                    self.display_welcome()
                    continue
                
                # Send message to API
                with console.status("[bold green]Thinking..."):
                    response = await self.client.send_message(user_input, self.session_id)
                
                # Update session ID
                self.session_id = response["session_id"]
                
                # Display response
                self.display_message("assistant", response["message"])
                
                # Display recommendations if any
                if response.get("recommendations"):
                    self.display_recommendations(response["recommendations"])
                
            except KeyboardInterrupt:
                console.print("\n\n[bold red]Interrupted! Exiting...[/bold red]")
                break
            except Exception as e:
                console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
                console.print("[dim]Try again or type 'help' for assistance.[/dim]")


async def quick_chat(message: str):
    """Quick one-off chat without interactive mode."""
    async with ApiClient() as client:
        try:
            # Check health first
            await client.health_check()
            
            # Send message
            with console.status("[bold green]Processing..."):
                response = await client.send_message(message)
            
            # Display response
            console.print(f"\n[bold green]Assistant:[/bold green] {response['message']}")
            
            # Display recommendations
            if response.get("recommendations"):
                interface = ChatInterface(client)
                interface.display_recommendations(response["recommendations"])
                
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


async def run_interactive():
    """Run interactive chat mode."""
    async with ApiClient() as client:
        try:
            # Check health first
            with console.status("[bold green]Connecting to service..."):
                await client.health_check()
            
            # Run chat interface
            interface = ChatInterface(client)
            await interface.run()
            
        except httpx.ConnectError:
            console.print("[bold red]Error:[/bold red] Cannot connect to the API service.")
            console.print("[dim]Make sure the service is running with: python -m app.main[/dim]")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")