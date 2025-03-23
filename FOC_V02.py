import tkinter as tk
from tkinter import ttk
import random

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")  # for Tkinter

##############################################################################
#                            SIMULATION LOGIC
##############################################################################

LIFESPAN = 85
GENERATIONS = 3
MONTE_CARLO_RUNS = 2000  # fewer or more as your system can handle
CHILDREN_PER_FAMILY = 2

DEFAULT_INCOME_MEAN = 35000
DEFAULT_INCOME_STD = 9000
DEFAULT_RETIREMENT_INCOME = 12000
DEFAULT_YOUTH_CONSUMPTION = 8000
DEFAULT_RETIREMENT_CONSUMPTION = 18000

def dynamic_consumption_rate(income, forced_saving_rate):
    """Progressive consumption minus forced saving."""
    if income < 30000:
        base_rate = 0.45
    elif income < 60000:
        base_rate = 0.55
    else:
        base_rate = 0.60

    c_rate = base_rate - forced_saving_rate
    return max(0.0, c_rate)

def progressive_inheritance_tax(inherited):
    """10% if < 300k, 20% if < 700k, else 35%."""
    if inherited < 300000:
        return 0.10
    elif inherited < 700000:
        return 0.20
    else:
        return 0.35

def dynamic_return_rate(asset, inflation):
    """Nominal return depends on asset size, minus inflation."""
    if asset < 100000:
        nominal = 0.04
    elif asset < 500000:
        nominal = 0.05
    else:
        nominal = 0.06
    return nominal - inflation

def simulate_three_generations(
    inflation=0.04,
    shock_prob=0.10,
    shock_impact=0.30,
    forced_saving=0.10,
    guarantee=10000,
    ubc=20000,
    n_runs=MONTE_CARLO_RUNS
):
    """Monte Carlo simulation of 3 generations."""
    INCOME_TAX = 0.30
    WEALTH_TAX_THRESHOLD = 300000
    WEALTH_TAX_RATE = 0.02

    generation_wealth = np.zeros((n_runs, GENERATIONS))

    for run in range(n_runs):
        assets_next_gen = 0

        for gen in range(GENERATIONS):
            age_assets = assets_next_gen

            for age in range(LIFESPAN):
                # Income & consumption
                if age < 22:
                    income = 0
                    consumption = DEFAULT_YOUTH_CONSUMPTION
                elif age < 65:
                    income = max(0, np.random.normal(DEFAULT_INCOME_MEAN, DEFAULT_INCOME_STD))
                    if np.random.rand() < shock_prob:
                        income *= (1 - shock_impact)
                    c_rate = dynamic_consumption_rate(income, forced_saving)
                    consumption = income * c_rate
                else:
                    income = DEFAULT_RETIREMENT_INCOME
                    consumption = DEFAULT_RETIREMENT_CONSUMPTION

                # Income tax
                taxed_income = income * (1 - INCOME_TAX)
                savings = taxed_income - consumption

                # UBC at 21
                if age == 21 and ubc > 0:
                    age_assets += ubc

                # Wealth tax
                if age_assets > WEALTH_TAX_THRESHOLD:
                    tax_amount = (age_assets - WEALTH_TAX_THRESHOLD) * 0.02
                    age_assets -= tax_amount

                # Returns
                r = dynamic_return_rate(age_assets, inflation)
                if np.random.rand() < shock_prob:
                    r *= (1 - shock_impact)

                age_assets = (age_assets + savings) * (1 + r)

            # End-of-life: inheritance
            inherited = age_assets / CHILDREN_PER_FAMILY
            i_tax_rate = progressive_inheritance_tax(inherited)
            inherited_after_tax = inherited * (1 - i_tax_rate)

            # Guarantee
            inherited_after_tax = max(inherited_after_tax, guarantee)

            generation_wealth[run, gen] = age_assets

            if gen < GENERATIONS - 1:
                assets_next_gen = inherited_after_tax

    return generation_wealth

def simulate_single_lifecycle(
    inflation=0.04,
    shock_prob=0.10,
    shock_impact=0.30,
    forced_saving=0.10,
    guarantee=10000,
    ubc=20000
):
    """Single generation (age 0..85) for plotting."""
    ages = np.arange(LIFESPAN)
    income_array = np.zeros(LIFESPAN)
    consumption_array = np.zeros(LIFESPAN)
    savings_array = np.zeros(LIFESPAN)
    assets_array = np.zeros(LIFESPAN)
    cumulative_savings_array = np.zeros(LIFESPAN)

    INCOME_TAX = 0.30
    WEALTH_TAX_THRESHOLD = 300000

    age_assets = 0
    cumulative = 0

    for i, age in enumerate(ages):
        if age < 22:
            income = 0
            consumption = DEFAULT_YOUTH_CONSUMPTION
        elif age < 65:
            income = max(0, np.random.normal(DEFAULT_INCOME_MEAN, DEFAULT_INCOME_STD))
            if np.random.rand() < shock_prob:
                income *= (1 - shock_impact)
            c_rate = dynamic_consumption_rate(income, forced_saving)
            consumption = income * c_rate
        else:
            income = DEFAULT_RETIREMENT_INCOME
            consumption = DEFAULT_RETIREMENT_CONSUMPTION

        taxed_income = income * (1 - INCOME_TAX)
        savings = taxed_income - consumption

        if age == 21 and ubc > 0:
            age_assets += ubc

        if age_assets > WEALTH_TAX_THRESHOLD:
            tax_amount = (age_assets - WEALTH_TAX_THRESHOLD) * 0.02
            age_assets -= tax_amount

        r = dynamic_return_rate(age_assets, inflation)
        if np.random.rand() < shock_prob:
            r *= (1 - shock_impact)

        age_assets = (age_assets + savings) * (1 + r)
        cumulative += savings

        income_array[i] = income
        consumption_array[i] = consumption
        savings_array[i] = savings
        assets_array[i] = age_assets
        cumulative_savings_array[i] = cumulative

    return ages, income_array, consumption_array, savings_array, assets_array, cumulative_savings_array


##############################################################################
#                        TKINTER APP WITH OPTIMIZATION
##############################################################################

class PolicyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("3-Gen Simulation with Simple Optimization")

        input_frame = ttk.Frame(self)
        input_frame.pack(pady=5, padx=5, fill="x")

        # ROW 0
        ttk.Label(input_frame, text="Inflation:").grid(row=0, column=0, sticky="e")
        self.inflation_var = tk.StringVar(value="0.05")
        ttk.Entry(input_frame, textvariable=self.inflation_var, width=6).grid(row=0, column=1, padx=5)
        self.opt_inflation_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(input_frame, text="Optimize?", variable=self.opt_inflation_var).grid(row=0, column=2)

        # ROW 1
        ttk.Label(input_frame, text="Shock Probability:").grid(row=1, column=0, sticky="e")
        self.shock_prob_var = tk.StringVar(value="0.15")
        ttk.Entry(input_frame, textvariable=self.shock_prob_var, width=6).grid(row=1, column=1, padx=5)
        self.opt_shock_prob_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(input_frame, text="Optimize?", variable=self.opt_shock_prob_var).grid(row=1, column=2)

        # ROW 2
        ttk.Label(input_frame, text="Shock Impact:").grid(row=2, column=0, sticky="e")
        self.shock_impact_var = tk.StringVar(value="0.3")
        ttk.Entry(input_frame, textvariable=self.shock_impact_var, width=6).grid(row=2, column=1, padx=5)
        self.opt_shock_impact_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(input_frame, text="Optimize?", variable=self.opt_shock_impact_var).grid(row=2, column=2)

        # ROW 3
        ttk.Label(input_frame, text="Forced Saving (0-0.3):").grid(row=3, column=0, sticky="e")
        self.forced_saving_var = tk.StringVar(value="0.1")
        ttk.Entry(input_frame, textvariable=self.forced_saving_var, width=6).grid(row=3, column=1, padx=5)
        self.opt_forced_saving_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(input_frame, text="Optimize?", variable=self.opt_forced_saving_var).grid(row=3, column=2)

        # ROW 4
        ttk.Label(input_frame, text="Guarantee (£):").grid(row=4, column=0, sticky="e")
        self.guarantee_var = tk.StringVar(value="10000")
        ttk.Entry(input_frame, textvariable=self.guarantee_var, width=6).grid(row=4, column=1, padx=5)
        self.opt_guarantee_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(input_frame, text="Optimize?", variable=self.opt_guarantee_var).grid(row=4, column=2)

        # ROW 5
        ttk.Label(input_frame, text="UBC (£):").grid(row=5, column=0, sticky="e")
        self.ubc_var = tk.StringVar(value="20000")
        ttk.Entry(input_frame, textvariable=self.ubc_var, width=6).grid(row=5, column=1, padx=5)
        self.opt_ubc_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(input_frame, text="Optimize?", variable=self.opt_ubc_var).grid(row=5, column=2)

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=5)
        run_button = ttk.Button(btn_frame, text="Run Simulation", command=self.run_simulation)
        run_button.grid(row=0, column=0, padx=10)
        opt_button = ttk.Button(btn_frame, text="Optimize", command=self.run_optimization)
        opt_button.grid(row=0, column=1, padx=10)

        # Results box
        self.results_box = tk.Text(self, width=60, height=12)
        self.results_box.pack(pady=5)

    def parse_inputs(self):
        """Parse user inputs (floats). Return dictionary of params."""
        params = {}
        try:
            params["inflation"] = float(self.inflation_var.get())
            params["shock_prob"] = float(self.shock_prob_var.get())
            params["shock_impact"] = float(self.shock_impact_var.get())
            params["forced_saving"] = float(self.forced_saving_var.get())
            params["guarantee"] = float(self.guarantee_var.get())
            params["ubc"] = float(self.ubc_var.get())
        except ValueError:
            self.results_box.insert(tk.END, "Invalid numeric input.\n")
            return None
        return params

    def run_simulation(self):
        """Run the simulation once with the user-provided parameters."""
        self.results_box.delete("1.0", tk.END)
        params = self.parse_inputs()
        if params is None:
            return

        # 1) Run the multi-generation simulation
        gen_wealth = simulate_three_generations(
            inflation=params["inflation"],
            shock_prob=params["shock_prob"],
            shock_impact=params["shock_impact"],
            forced_saving=params["forced_saving"],
            guarantee=params["guarantee"],
            ubc=params["ubc"]
        )
        avg_wealth_by_gen = gen_wealth.mean(axis=0)

        self.results_box.insert(tk.END, "===== 3-Gen Simulation Results =====\n")
        for i, val in enumerate(avg_wealth_by_gen, start=1):
            self.results_box.insert(tk.END, f"Generation {i}: Avg Final Wealth = £{val:,.2f}\n")

        # 2) Plot 3-generation
        plt.figure(figsize=(7,4))
        plt.plot(range(1, GENERATIONS+1), avg_wealth_by_gen, marker='o', color='orange')
        plt.title("Average Final Wealth Over 3 Generations")
        plt.xlabel("Generation")
        plt.ylabel("Wealth (£)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # 3) Single-lifecycle example
        (ages, inc, cons, sav, ast, cum_sav) = simulate_single_lifecycle(
            inflation=params["inflation"],
            shock_prob=params["shock_prob"],
            shock_impact=params["shock_impact"],
            forced_saving=params["forced_saving"],
            guarantee=params["guarantee"],
            ubc=params["ubc"]
        )

        plt.figure(figsize=(10,6))
        plt.plot(ages, ast, label='Assets', color='orange')
        plt.plot(ages, cum_sav, label='Cumulative Savings', color='orangered')
        plt.plot(ages, inc, label='Income', color='deeppink')
        plt.plot(ages, cons, label='Consumption', color='hotpink')
        plt.axvline(x=21, color='green', linestyle='--', label='UBC at 21')
        plt.axvline(x=65, color='gray', linestyle='--', label='Retirement')

        plt.title("Single Lifecycle: Income, Savings, Consumption, Assets")
        plt.xlabel("Age")
        plt.ylabel("GBP (£)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def run_optimization(self):
        """
        Simple random search over selected (checked) parameters.
        We'll do e.g. 20 random samples. Then pick the scenario that
        maximizes average final wealth in Generation 3.
        """
        self.results_box.delete("1.0", tk.END)
        base_params = self.parse_inputs()
        if base_params is None:
            return

        # Define search ranges for each param
        # You can tweak these as you like
        infl_range = (0.0, 0.1)        # inflation
        shockp_range = (0.0, 0.5)      # shock_prob
        shocki_range = (0.1, 0.5)      # shock_impact
        forced_range = (0.0, 0.3)      # forced_saving
        guar_range = (0.0, 50000)      # guarantee
        ubc_range = (0.0, 50000)       # ubc

        N_SAMPLES = 20

        best_score = -1e15
        best_params = None
        best_wealth = None

        for i in range(N_SAMPLES):
            # Start with the user's inputs
            trial = dict(base_params)

            # If user checked "Optimize inflation," sample from infl_range
            if self.opt_inflation_var.get():
                trial["inflation"] = random.uniform(*infl_range)

            # If user checked "Optimize shock_prob," sample from shockp_range
            if self.opt_shock_prob_var.get():
                trial["shock_prob"] = random.uniform(*shockp_range)

            # If user checked "Optimize shock_impact," sample from shocki_range
            if self.opt_shock_impact_var.get():
                trial["shock_impact"] = random.uniform(*shocki_range)

            # If user checked "Optimize forced_saving," sample from forced_range
            if self.opt_forced_saving_var.get():
                trial["forced_saving"] = random.uniform(*forced_range)

            # If user checked "Optimize guarantee," sample from guar_range
            if self.opt_guarantee_var.get():
                trial["guarantee"] = random.uniform(*guar_range)

            # If user checked "Optimize ubc," sample from ubc_range
            if self.opt_ubc_var.get():
                trial["ubc"] = random.uniform(*ubc_range)

            # Run simulation for this trial
            gen_wealth = simulate_three_generations(
                inflation=trial["inflation"],
                shock_prob=trial["shock_prob"],
                shock_impact=trial["shock_impact"],
                forced_saving=trial["forced_saving"],
                guarantee=trial["guarantee"],
                ubc=trial["ubc"]
            )
            avg_wealth_by_gen = gen_wealth.mean(axis=0)

            # We'll pick the scenario that yields the highest final wealth in Gen3
            score = avg_wealth_by_gen[-1]  # generation 3

            if score > best_score:
                best_score = score
                best_params = trial
                best_wealth = avg_wealth_by_gen

        # Done searching
        self.results_box.insert(tk.END, "=== BEST SCENARIO FOUND ===\n")
        if best_params is None:
            self.results_box.insert(tk.END, "No valid scenario.\n")
            return

        # Show best scenario
        for k, v in best_params.items():
            self.results_box.insert(tk.END, f"{k}: {v:.4f}\n")
        self.results_box.insert(tk.END, "\nAverage Final Wealth by Gen:\n")
        for i, val in enumerate(best_wealth, start=1):
            self.results_box.insert(tk.END, f" Gen {i}: £{val:,.2f}\n")
        self.results_box.insert(tk.END, f"\nObjective (Gen3 Wealth) = £{best_score:,.2f}\n")

        # Now produce plots for best scenario
        plt.figure(figsize=(7,4))
        plt.plot(range(1, GENERATIONS+1), best_wealth, marker='o', color='orange')
        plt.title("BEST Scenario: Avg Final Wealth Over 3 Generations")
        plt.xlabel("Generation")
        plt.ylabel("Wealth (£)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Single-lifecycle for best scenario
        (ages, inc, cons, sav, ast, cum_sav) = simulate_single_lifecycle(
            inflation=best_params["inflation"],
            shock_prob=best_params["shock_prob"],
            shock_impact=best_params["shock_impact"],
            forced_saving=best_params["forced_saving"],
            guarantee=best_params["guarantee"],
            ubc=best_params["ubc"]
        )
        plt.figure(figsize=(10,6))
        plt.plot(ages, ast, label='Assets', color='orange')
        plt.plot(ages, cum_sav, label='Cumulative Savings', color='orangered')
        plt.plot(ages, inc, label='Income', color='deeppink')
        plt.plot(ages, cons, label='Consumption', color='hotpink')
        plt.axvline(x=21, color='green', linestyle='--', label='UBC at 21')
        plt.axvline(x=65, color='gray', linestyle='--', label='Retirement')

        plt.title("BEST Scenario Lifecycle")
        plt.xlabel("Age")
        plt.ylabel("GBP (£)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main():
    app = PolicyApp()
    app.mainloop()

if __name__ == "__main__":
    main()
