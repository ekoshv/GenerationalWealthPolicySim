import tkinter as tk
from tkinter import ttk
import random
import numpy as np
import matplotlib.pyplot as plt
import pygad
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------------------------
# Utility Functions
# ---------------------------
def compute_gini(x):
    """Compute Gini coefficient for an array of values."""
    x = np.array(x)
    if np.amin(x) < 0:
        x -= np.amin(x)
    x += 1e-10  # avoid division by zero
    x_sorted = np.sort(x)
    n = x.shape[0]
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n

def clear_frame(frame):
    """Remove all widgets from a frame."""
    for widget in frame.winfo_children():
        widget.destroy()

# ---------------------------
# Tax Policy (Fixed Income Classes)
# ---------------------------
class TaxPolicy:
    def __init__(self, lower_rate=0.10, middle_rate=0.20, high_rate=0.30,
                 lower_threshold=30000, middle_threshold=70000):
        """
        Tax rates as decimals. Thresholds are fixed so that inflation does not alter class boundaries.
        """
        self.lower_rate = lower_rate
        self.middle_rate = middle_rate
        self.high_rate = high_rate
        self.lower_threshold = lower_threshold
        self.middle_threshold = middle_threshold

    def compute_tax(self, income):
        """Compute tax based on income class."""
        if income < self.lower_threshold:
            return income * self.lower_rate
        elif income < self.middle_threshold:
            return income * self.middle_rate
        else:
            return income * self.high_rate

# ---------------------------
# Government (Dynamic Fiscal Policy)
# ---------------------------
class Government:
    def __init__(self, tax_policy: TaxPolicy, ubc=20000, guarantee=10000):
        """
        ubc: Universal Basic Capital given at age 21.
        guarantee: Minimum inheritance value guaranteed.
        """
        self.tax_policy = tax_policy
        self.ubc = ubc
        self.guarantee = guarantee
        self.revenue = 0.0
        self.cost = 0.0

    def reset(self):
        self.revenue = 0.0
        self.cost = 0.0

    def collect_tax(self, income):
        tax = self.tax_policy.compute_tax(income)
        self.revenue += tax
        return tax

    def pay_ubc(self):
        self.cost += self.ubc
        return self.ubc

    def pay_guarantee_shortfall(self, shortfall):
        self.cost += shortfall
        return shortfall

    def fiscal_balance(self):
        return self.revenue - self.cost

    def update_policy(self):
        # Example dynamic adjustment: if balance is negative, raise tax rates slightly.
        if self.fiscal_balance() < 0:
            self.tax_policy.lower_rate *= 1.01
            self.tax_policy.middle_rate *= 1.01
            self.tax_policy.high_rate *= 1.01

# ---------------------------
# Multi-Generation Simulation
# ---------------------------
def simulate_three_generations(
    inflation=0.05,
    shock_prob=0.15,
    shock_impact=0.30,
    forced_saving=0.10,
    guarantee=10000,
    ubc=20000,
    n_children=2,
    wage_growth=0.01,
    unemployment_prob=0.05,
    unemployment_benefit=5000,
    base_interest=0.02,
    interest_trend=0.0,
    productivity_growth=0.00,
    old_age_medical=2000,
    housing_choice=False,
    mortgage_payment=5000,
    stocks_fraction=0.5,
    overspending_prob=0.05,
    overspending_factor=1.2,
    n_runs=2000
):
    """
    Simulate three generations of a household's lifecycle.
    Returns:
      - generation_wealth: (n_runs x 3) array of final wealth per generation.
      - total government cost and total government revenue.
    """
    GENERATIONS = 3
    LIFESPAN = 85
    DEFAULT_INCOME_MEAN = 35000
    DEFAULT_INCOME_STD = 8000
    DEFAULT_RETIREMENT_INCOME = 12000
    DEFAULT_YOUTH_CONSUMPTION = 8000
    DEFAULT_RETIREMENT_CONSUMPTION = 18000

    generation_wealth = np.zeros((n_runs, GENERATIONS))
    government_cost = 0.0
    government_revenue = 0.0

    for run in range(n_runs):
        assets_next_gen = 0
        for gen in range(GENERATIONS):
            age_assets = assets_next_gen
            yearly_income_mean = DEFAULT_INCOME_MEAN

            for age in range(LIFESPAN):
                if age < 22:
                    income = 0
                    consumption = DEFAULT_YOUTH_CONSUMPTION
                    tax = 0  # No income, no tax.
                    government_revenue += tax
                    net_income = income - tax
                elif age < 65:
                    yearly_income_mean *= (1 + wage_growth)
                    yearly_income_mean *= (1 + productivity_growth * (age / LIFESPAN))
                    raw_income = max(0, np.random.normal(yearly_income_mean, DEFAULT_INCOME_STD))
                    if np.random.rand() < unemployment_prob:
                        income = unemployment_benefit
                    else:
                        income = raw_income
                    tax = (income * 0.20) if income < 60000 else (income * 0.30)
                    government_revenue += tax
                    net_income = income - tax

                    base_rate = 0.55 if income < 60000 else 0.60
                    c_rate = max(0, base_rate - forced_saving)
                    consumption = income * c_rate
                    if np.random.rand() < overspending_prob:
                        consumption *= overspending_factor
                    if housing_choice:
                        consumption += mortgage_payment
                else:
                    income = DEFAULT_RETIREMENT_INCOME
                    consumption = DEFAULT_RETIREMENT_CONSUMPTION + old_age_medical
                    tax = (income * 0.20)
                    government_revenue += tax
                    net_income = income - tax

                savings = net_income - consumption

                if age == 21 and ubc > 0:
                    age_assets += ubc
                    government_cost += ubc

                if age_assets > 300000:
                    tax_w = (age_assets - 300000) * 0.02
                    age_assets -= tax_w
                    government_revenue += tax_w

                curr_interest = base_interest + age * interest_trend
                stock_nominal = curr_interest + 0.03
                bond_nominal = curr_interest
                nominal = stocks_fraction * stock_nominal + (1 - stocks_fraction) * bond_nominal
                real_return = nominal - inflation
                if np.random.rand() < shock_prob:
                    real_return *= (1 - shock_impact)
                age_assets = (age_assets + savings) * (1 + real_return)

            inherited = age_assets / n_children
            i_tax_rate = 0.10 if inherited < 300000 else (0.20 if inherited < 700000 else 0.35)
            tax_inheritance = inherited * i_tax_rate
            government_revenue += tax_inheritance
            inherited_after_tax = inherited * (1 - i_tax_rate)
            if inherited_after_tax < guarantee:
                shortfall = guarantee - inherited_after_tax
                government_cost += shortfall
                inherited_after_tax = guarantee

            generation_wealth[run, gen] = age_assets
            if gen < GENERATIONS - 1:
                assets_next_gen = inherited_after_tax

    return generation_wealth, government_cost, government_revenue

# ---------------------------
# Single Lifecycle Simulation (for Life-Cycle Plots)
# ---------------------------
def simulate_single_lifecycle(
    inflation=0.05,
    shock_prob=0.15,
    shock_impact=0.30,
    forced_saving=0.10,
    guarantee=10000,
    ubc=20000,
    n_children=2,
    wage_growth=0.01,
    unemployment_prob=0.05,
    unemployment_benefit=5000,
    base_interest=0.02,
    interest_trend=0.0,
    productivity_growth=0.00,
    old_age_medical=2000,
    housing_choice=False,
    mortgage_payment=5000,
    stocks_fraction=0.5,
    overspending_prob=0.05,
    overspending_factor=1.2
):
    LIFESPAN = 85
    ages = np.arange(LIFESPAN)
    income_array = np.zeros(LIFESPAN)
    consumption_array = np.zeros(LIFESPAN)
    savings_array = np.zeros(LIFESPAN)
    assets_array = np.zeros(LIFESPAN)
    cumulative_savings_array = np.zeros(LIFESPAN)
    age_assets = 0
    yearly_income_mean = 35000
    DEFAULT_YOUTH_CONSUMPTION = 8000
    DEFAULT_RETIREMENT_CONSUMPTION = 18000
    DEFAULT_RETIREMENT_INCOME = 12000
    DEFAULT_INCOME_STD = 8000

    for i, age in enumerate(ages):
        if age < 22:
            income = 0
            consumption = DEFAULT_YOUTH_CONSUMPTION
            net_income = 0
        elif age < 65:
            yearly_income_mean *= (1 + wage_growth)
            yearly_income_mean *= (1 + productivity_growth * (age / LIFESPAN))
            raw_income = max(0, np.random.normal(yearly_income_mean, DEFAULT_INCOME_STD))
            if np.random.rand() < unemployment_prob:
                income = unemployment_benefit
            else:
                income = raw_income
            net_income = income - (income * 0.20)
            base_rate = 0.55 if income < 60000 else 0.60
            consumption = income * (base_rate - forced_saving)
        else:
            income = DEFAULT_RETIREMENT_INCOME
            consumption = DEFAULT_RETIREMENT_CONSUMPTION + old_age_medical
            net_income = income - (income * 0.20)
        savings = net_income - consumption
        if age == 21 and ubc > 0:
            age_assets += ubc
        if age_assets > 300000:
            age_assets -= (age_assets - 300000) * 0.02
        curr_interest = base_interest + i * interest_trend
        stock_nominal = curr_interest + 0.03
        bond_nominal = curr_interest
        nominal = stocks_fraction * stock_nominal + (1 - stocks_fraction) * bond_nominal
        real_return = nominal - inflation
        if np.random.rand() < shock_prob:
            real_return *= (1 - shock_impact)
        age_assets = (age_assets + savings) * (1 + real_return)
        income_array[i] = income
        consumption_array[i] = consumption
        savings_array[i] = savings
        assets_array[i] = age_assets
        if i == 0:
            cumulative_savings_array[i] = savings
        else:
            cumulative_savings_array[i] = cumulative_savings_array[i - 1] + savings

    return ages, income_array, consumption_array, savings_array, assets_array, cumulative_savings_array

# ---------------------------
# Agent-Based Simulation
# ---------------------------
class Agent:
    def __init__(self, initial_assets=0.0, income_mean=35000):
        self.assets = initial_assets
        self.income_mean = income_mean
        self.age = 20
        self.alive = True
        self.history = []  # Records of (age, assets)

    def simulate_year(self, government: Government, unemployment_rate=0.05,
                      unemployment_benefit=5000, forced_saving=0.10,
                      wage_growth=0.01, inflation=0.02):
        if np.random.rand() < unemployment_rate:
            income = unemployment_benefit
        else:
            income = max(0, np.random.normal(self.income_mean, 8000))
        tax = government.collect_tax(income)
        net_income = income - tax
        consumption = net_income * (0.55 - forced_saving)
        savings = net_income - consumption
        real_return = 0.02 - inflation
        self.assets = (self.assets + savings) * (1 + real_return)
        self.history.append((self.age, self.assets))
        self.age += 1
        return income, consumption, savings, self.assets

def simulate_population(num_agents=1000, num_years=40, government: Government = None,
                        unemployment_rate=0.05, forced_saving=0.10, wage_growth=0.01,
                        inflation=0.02):
    agents = [Agent(initial_assets=random.uniform(0, 10000)) for _ in range(num_agents)]
    if government is not None:
        government.reset()
    for _ in range(num_years):
        for agent in agents:
            if agent.alive:
                agent.simulate_year(government, unemployment_rate, 5000, forced_saving, wage_growth, inflation)
    wealths = np.array([agent.assets for agent in agents])
    gini = compute_gini(wealths)
    avg_wealth = np.mean(wealths)
    fiscal_balance = government.fiscal_balance() if government is not None else None
    return wealths, gini, avg_wealth, fiscal_balance

# ---------------------------
# GUI: Notebook-Based Society Simulator
# ---------------------------
class SocietySimulatorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Advanced Society Simulator")
        self.geometry("1150x750")
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)

        # Create three tabs.
        self.tab_multi = ttk.Frame(self.notebook)
        self.tab_agent = ttk.Frame(self.notebook)
        self.tab_ga = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_multi, text="Multi-Generation Simulation")
        self.notebook.add(self.tab_agent, text="Agent-Based Simulation")
        self.notebook.add(self.tab_ga, text="Policy Optimization")

        self.setup_multi_tab()
        self.setup_agent_tab()
        self.setup_ga_tab()

    # ----- Multi-Generation Simulation Tab -----
    def setup_multi_tab(self):
        # Split the tab into left (controls & text) and right (plots) frames.
        left_frame = ttk.Frame(self.tab_multi)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        right_frame = ttk.Frame(self.tab_multi)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.tab_multi.columnconfigure(0, weight=1)
        self.tab_multi.columnconfigure(1, weight=2)

        header = ttk.Label(left_frame, text="Multi-Generation Simulation Parameters", font=("Arial", 14))
        header.pack(pady=10)
        self.multi_frame = ttk.Frame(left_frame)
        self.multi_frame.pack(pady=10)
        params = [
            ("Inflation", "0.05"),
            ("ShockProbability", "0.15"),
            ("ShockImpact", "0.3"),
            ("ForcedSaving", "0.1"),
            ("Guarantee", "10000"),
            ("UBC", "20000"),
            ("ChildrenPerFamily", "2"),
            ("WageGrowth", "0.01"),
            ("UnemploymentProb", "0.05"),
            ("UnemploymentBenefit", "5000"),
            ("BaseInterest", "0.02"),
            ("InterestTrend", "0.0"),
            ("ProductivityGrowth", "0.0"),
            ("OldAgeMedical", "2000"),
            ("MortgagePayment", "5000"),
            ("StocksFraction", "0.5"),
            ("OverspendingProb", "0.05"),
            ("OverspendingFactor", "1.2"),
            ("MonteCarloRuns", "2000")
        ]
        self.multi_params = {}
        row = 0
        for label, default in params:
            ttk.Label(self.multi_frame, text=label+":").grid(row=row, column=0, sticky="e", padx=5, pady=2)
            var = tk.StringVar(value=default)
            ttk.Entry(self.multi_frame, textvariable=var, width=10).grid(row=row, column=1, padx=5, pady=2)
            self.multi_params[label] = var
            row += 1

        run_button = ttk.Button(left_frame, text="Run Multi-Generation Simulation", command=self.run_multi_sim)
        run_button.pack(pady=10)
        self.multi_text = tk.Text(left_frame, width=50, height=8)
        self.multi_text.pack(pady=10)
        self.multi_plot_frame = right_frame  # Right frame reserved for plots.

    def parse_multi_params(self):
        p = {}
        for key, var in self.multi_params.items():
            try:
                if key in ["MonteCarloRuns", "ChildrenPerFamily"]:
                    p[key] = int(var.get())
                else:
                    p[key] = float(var.get())
            except:
                p[key] = 0.0
        return p

    def run_multi_sim(self):
        self.multi_text.delete("1.0", tk.END)
        p = self.parse_multi_params()
        gen_wealth, gov_cost, gov_rev = simulate_three_generations(
            inflation=p["Inflation"],
            shock_prob=p["ShockProbability"],
            shock_impact=p["ShockImpact"],
            forced_saving=p["ForcedSaving"],
            guarantee=p["Guarantee"],
            ubc=p["UBC"],
            n_children=p["ChildrenPerFamily"],
            wage_growth=p["WageGrowth"],
            unemployment_prob=p["UnemploymentProb"],
            unemployment_benefit=p["UnemploymentBenefit"],
            base_interest=p["BaseInterest"],
            interest_trend=p["InterestTrend"],
            productivity_growth=p["ProductivityGrowth"],
            old_age_medical=p["OldAgeMedical"],
            mortgage_payment=p["MortgagePayment"],
            stocks_fraction=p["StocksFraction"],
            overspending_prob=p["OverspendingProb"],
            overspending_factor=p["OverspendingFactor"],
            n_runs=p["MonteCarloRuns"]
        )
        avg_wealth = gen_wealth.mean(axis=0)
        avg_gov_cost = gov_cost / p["MonteCarloRuns"]
        avg_gov_rev = gov_rev / p["MonteCarloRuns"]
        gov_balance = avg_gov_rev - avg_gov_cost

        self.multi_text.insert(tk.END, "=== Multi-Generation Simulation Results ===\n")
        for i, val in enumerate(avg_wealth, start=1):
            self.multi_text.insert(tk.END, f"Generation {i} Average Wealth: £{val:,.2f}\n")
        self.multi_text.insert(tk.END, f"\nAvg Government Cost: £{avg_gov_cost:,.2f}\n")
        self.multi_text.insert(tk.END, f"Avg Government Revenue: £{avg_gov_rev:,.2f}\n")
        self.multi_text.insert(tk.END, f"Avg Government Fiscal Balance: £{gov_balance:,.2f}\n")

        clear_frame(self.multi_plot_frame)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot([1, 2, 3], avg_wealth, marker='o', color='orange')
        ax1.set_title("Avg Final Wealth per Generation")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Wealth (£)")
        ax1.grid(True)
        ages, inc, cons, sav, ast, cum_sav = simulate_single_lifecycle(
            inflation=p["Inflation"],
            shock_prob=p["ShockProbability"],
            shock_impact=p["ShockImpact"],
            forced_saving=p["ForcedSaving"],
            guarantee=p["Guarantee"],
            ubc=p["UBC"],
            n_children=p["ChildrenPerFamily"],
            wage_growth=p["WageGrowth"],
            unemployment_prob=p["UnemploymentProb"],
            unemployment_benefit=p["UnemploymentBenefit"],
            base_interest=p["BaseInterest"],
            interest_trend=p["InterestTrend"],
            productivity_growth=p["ProductivityGrowth"],
            old_age_medical=p["OldAgeMedical"],
            mortgage_payment=p["MortgagePayment"],
            stocks_fraction=p["StocksFraction"],
            overspending_prob=p["OverspendingProb"],
            overspending_factor=p["OverspendingFactor"]
        )
        ax2.plot(ages, ast, label='Assets', color='orange')
        ax2.plot(ages, cum_sav, label='Cumulative Savings', color='red')
        ax2.plot(ages, inc, label='Income', color='deeppink')
        ax2.plot(ages, cons, label='Consumption', color='hotpink')
        ax2.axvline(x=21, color='green', linestyle='--', label='UBC@21')
        ax2.axvline(x=65, color='gray', linestyle='--', label='Retire@65')
        ax2.set_title("Single Lifecycle")
        ax2.set_xlabel("Age")
        ax2.set_ylabel("Wealth/Income (£)")
        ax2.legend(fontsize=8)
        ax2.grid(True)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.multi_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ----- Agent-Based Simulation Tab -----
    def setup_agent_tab(self):
        left_frame = ttk.Frame(self.tab_agent)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        right_frame = ttk.Frame(self.tab_agent)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.tab_agent.columnconfigure(0, weight=1)
        self.tab_agent.columnconfigure(1, weight=2)

        header = ttk.Label(left_frame, text="Agent-Based Simulation Parameters", font=("Arial", 14))
        header.pack(pady=10)
        frame = ttk.Frame(left_frame)
        frame.pack(pady=10)
        ttk.Label(frame, text="Number of Agents:").grid(row=0, column=0, sticky="e")
        self.num_agents_var = tk.StringVar(value="1000")
        ttk.Entry(frame, textvariable=self.num_agents_var, width=8).grid(row=0, column=1, padx=5)
        ttk.Label(frame, text="Number of Years:").grid(row=1, column=0, sticky="e")
        self.num_years_var = tk.StringVar(value="40")
        ttk.Entry(frame, textvariable=self.num_years_var, width=8).grid(row=1, column=1, padx=5)
        ttk.Label(frame, text="Unemployment Rate:").grid(row=2, column=0, sticky="e")
        self.unemp_rate_var = tk.StringVar(value="0.05")
        ttk.Entry(frame, textvariable=self.unemp_rate_var, width=8).grid(row=2, column=1, padx=5)
        run_agent_button = ttk.Button(left_frame, text="Run Agent-Based Simulation", command=self.run_agent_simulation)
        run_agent_button.pack(pady=10)
        self.agent_text = tk.Text(left_frame, width=50, height=6)
        self.agent_text.pack(pady=10)
        self.agent_plot_frame = right_frame  # Right frame for plots.

    def run_agent_simulation(self):
        try:
            num_agents = int(self.num_agents_var.get())
            num_years = int(self.num_years_var.get())
            unemp_rate = float(self.unemp_rate_var.get())
        except:
            self.agent_text.insert(tk.END, "Invalid input parameters.\n")
            return
        tax_policy = TaxPolicy(lower_rate=0.10, middle_rate=0.20, high_rate=0.30,
                               lower_threshold=30000, middle_threshold=70000)
        government = Government(tax_policy=tax_policy, ubc=20000, guarantee=10000)
        wealths, gini, avg_wealth, fiscal_balance = simulate_population(
            num_agents=num_agents, num_years=num_years, government=government,
            unemployment_rate=unemp_rate, forced_saving=0.10, wage_growth=0.01, inflation=0.02)
        self.agent_text.delete("1.0", tk.END)
        self.agent_text.insert(tk.END, "Agent-Based Simulation Results:\n")
        self.agent_text.insert(tk.END, f"Average Wealth: £{avg_wealth:,.2f}\n")
        self.agent_text.insert(tk.END, f"Gini Coefficient: {gini:.4f}\n")
        self.agent_text.insert(tk.END, f"Government Fiscal Balance: £{fiscal_balance:,.2f}\n")
        clear_frame(self.agent_plot_frame)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.hist(wealths, bins=50, color="skyblue", edgecolor="black")
        ax.set_title("Wealth Distribution")
        ax.set_xlabel("Wealth (£)")
        ax.set_ylabel("Number of Agents")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.agent_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    # ----- Policy Optimization (GA) Tab -----
    def setup_ga_tab(self):
        left_frame = ttk.Frame(self.tab_ga)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        right_frame = ttk.Frame(self.tab_ga)
        right_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.tab_ga.columnconfigure(0, weight=1)
        self.tab_ga.columnconfigure(1, weight=2)

        header = ttk.Label(left_frame, text="Policy Optimization via GA", font=("Arial", 14))
        header.pack(pady=10)
        frame = ttk.Frame(left_frame)
        frame.pack(pady=10)
        ttk.Label(frame, text="Lower Income Tax Rate:").grid(row=0, column=0, sticky="e")
        self.lower_tax_var = tk.StringVar(value="0.10")
        ttk.Entry(frame, textvariable=self.lower_tax_var, width=8).grid(row=0, column=1, padx=5)
        ttk.Label(frame, text="Middle Income Tax Rate:").grid(row=1, column=0, sticky="e")
        self.middle_tax_var = tk.StringVar(value="0.20")
        ttk.Entry(frame, textvariable=self.middle_tax_var, width=8).grid(row=1, column=1, padx=5)
        ttk.Label(frame, text="High Income Tax Rate:").grid(row=2, column=0, sticky="e")
        self.high_tax_var = tk.StringVar(value="0.30")
        ttk.Entry(frame, textvariable=self.high_tax_var, width=8).grid(row=2, column=1, padx=5)
        ttk.Label(frame, text="UBC:").grid(row=3, column=0, sticky="e")
        self.ubc_ga_var = tk.StringVar(value="20000")
        ttk.Entry(frame, textvariable=self.ubc_ga_var, width=8).grid(row=3, column=1, padx=5)
        ttk.Label(frame, text="Guarantee:").grid(row=4, column=0, sticky="e")
        self.guarantee_ga_var = tk.StringVar(value="10000")
        ttk.Entry(frame, textvariable=self.guarantee_ga_var, width=8).grid(row=4, column=1, padx=5)
        run_ga_button = ttk.Button(left_frame, text="Run GA Optimization", command=self.run_ga_optimization)
        run_ga_button.pack(pady=10)
        self.ga_text = tk.Text(left_frame, width=50, height=6)
        self.ga_text.pack(pady=10)
        self.ga_plot_frame = right_frame  # Right frame for plots.

    def run_ga_optimization(self):
        try:
            lower_rate = float(self.lower_tax_var.get())
            middle_rate = float(self.middle_tax_var.get())
            high_rate = float(self.high_tax_var.get())
            ubc_val = float(self.ubc_ga_var.get())
            guarantee_val = float(self.guarantee_ga_var.get())
        except:
            self.ga_text.insert(tk.END, "Invalid GA parameter inputs.\n")
            return

        gene_space = [
            {"low": 0.05, "high": 0.15},    # Lower income tax rate.
            {"low": 0.15, "high": 0.30},    # Middle income tax rate.
            {"low": 0.25, "high": 0.50},    # High income tax rate.
            {"low": ubc_val * 0.5, "high": ubc_val * 1.5},   # UBC.
            {"low": guarantee_val * 0.5, "high": guarantee_val * 1.5}  # Guarantee.
        ]

        def fitness_func(ga_instance, solution, solution_idx):
            lower_rate_ga, middle_rate_ga, high_rate_ga, ubc_ga, guarantee_ga = solution
            tax_policy = TaxPolicy(lower_rate=lower_rate_ga, middle_rate=middle_rate_ga,
                                   high_rate=high_rate_ga, lower_threshold=30000, middle_threshold=70000)
            government = Government(tax_policy=tax_policy, ubc=ubc_ga, guarantee=guarantee_ga)
            wealths, gini, avg_wealth, fiscal_balance = simulate_population(
                num_agents=500, num_years=30, government=government,
                unemployment_rate=0.05, forced_saving=0.10, wage_growth=0.01, inflation=0.02)
            if fiscal_balance < 0 or avg_wealth <= 0:
                return -1e6
            fitness = (0.5 - avg_wealth / 1e6) + (fiscal_balance / 1e5) - gini
            return fitness

        ga_instance = pygad.GA(
            num_generations=30,
            num_parents_mating=10,
            fitness_func=fitness_func,
            sol_per_pop=20,
            num_genes=5,
            gene_space=gene_space,
            mutation_percent_genes=20,
            mutation_num_genes=1,
            mutation_type="random",
            crossover_type="single_point",
            stop_criteria="saturate_10"
        )

        self.ga_text.insert(tk.END, "Running GA Optimization...\n")
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        lower_rate_ga, middle_rate_ga, high_rate_ga, ubc_ga, guarantee_ga = solution

        tax_policy = TaxPolicy(lower_rate=lower_rate_ga, middle_rate=middle_rate_ga,
                               high_rate=high_rate_ga, lower_threshold=30000, middle_threshold=70000)
        government = Government(tax_policy=tax_policy, ubc=ubc_ga, guarantee=guarantee_ga)
        wealths, gini, avg_wealth, fiscal_balance = simulate_population(
            num_agents=500, num_years=30, government=government,
            unemployment_rate=0.05, forced_saving=0.10, wage_growth=0.01, inflation=0.02)
        self.ga_text.delete("1.0", tk.END)
        self.ga_text.insert(tk.END, "=== GA Optimization Result ===\n")
        self.ga_text.insert(tk.END, f"Best Fitness: {solution_fitness:.4f}\n")
        self.ga_text.insert(tk.END, f"Optimized Lower Tax Rate: {lower_rate_ga:.4f}\n")
        self.ga_text.insert(tk.END, f"Optimized Middle Tax Rate: {middle_rate_ga:.4f}\n")
        self.ga_text.insert(tk.END, f"Optimized High Tax Rate: {high_rate_ga:.4f}\n")
        self.ga_text.insert(tk.END, f"Optimized UBC: £{ubc_ga:,.2f}\n")
        self.ga_text.insert(tk.END, f"Optimized Guarantee: £{guarantee_ga:,.2f}\n")
        self.ga_text.insert(tk.END, f"Average Wealth: £{avg_wealth:,.2f}\n")
        self.ga_text.insert(tk.END, f"Fiscal Balance: £{fiscal_balance:,.2f}\n")
        self.ga_text.insert(tk.END, f"Gini Coefficient: {gini:.4f}\n")
        clear_frame(self.ga_plot_frame)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.hist(wealths, bins=50, color="skyblue", edgecolor="black")
        ax.set_title("Wealth Distribution under Optimized Policy")
        ax.set_xlabel("Wealth (£)")
        ax.set_ylabel("Number of Agents")
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.ga_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

def main():
    app = SocietySimulatorApp()
    app.mainloop()

if __name__ == "__main__":
    main()
