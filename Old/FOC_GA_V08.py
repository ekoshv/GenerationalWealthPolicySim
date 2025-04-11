import tkinter as tk
from tkinter import ttk
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pygad  # ensure pygad is installed: pip install pygad

matplotlib.use("TkAgg")  # For Tkinter

##############################################################################
#                          ADVANCED ECONOMIC LOGIC
##############################################################################

LIFESPAN = 85
GENERATIONS = 3
MONTE_CARLO_RUNS = 2000

DEFAULT_INCOME_MEAN = 35000
DEFAULT_INCOME_STD = 8000
DEFAULT_RETIREMENT_INCOME = 12000
DEFAULT_YOUTH_CONSUMPTION = 8000
DEFAULT_RETIREMENT_CONSUMPTION = 18000

INCOME_TAX_BRACKETS = [
    (30000, 0.20),
    (60000, 0.30),
    (999999999, 0.40)
]

WEALTH_TAX_THRESHOLD = 300000
WEALTH_TAX_RATE = 0.02

def bracket_income_tax(income):
    """Compute income tax based on simplified brackets."""
    remaining = income
    total_tax = 0
    lower_bound = 0
    for upper_bound, rate in INCOME_TAX_BRACKETS:
        taxable = min(remaining, upper_bound - lower_bound)
        if taxable > 0:
            total_tax += taxable * rate
            remaining -= taxable
        lower_bound = upper_bound
        if remaining <= 0:
            break
    return total_tax

def progressive_inheritance_tax(inherited):
    """Return the inheritance tax rate."""
    if inherited < 300000:
        return 0.10
    elif inherited < 700000:
        return 0.20
    else:
        return 0.35

def apply_asset_allocation(asset, stocks_fraction, base_interest, inflation, shock_prob, shock_impact):
    """
    Calculate the weighted nominal return minus inflation plus a potential random shock.
    """
    stock_nominal = base_interest + 0.03
    bond_nominal  = base_interest
    nominal = stocks_fraction * stock_nominal + (1 - stocks_fraction) * bond_nominal
    real_return = nominal - inflation
    if np.random.rand() < shock_prob:
        real_return *= (1 - shock_impact)
    return real_return

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
    n_runs=MONTE_CARLO_RUNS
):
    """
    Simulate three generations of a household’s lifecycle.
    Returns:
      - generation_wealth: (n_runs x GENERATIONS array) final wealth per generation.
      - government_cost: total government outlays (UBC and guarantee shortfalls).
      - government_revenue: total tax revenue (income, wealth, inheritance).
    """
    generation_wealth = np.zeros((n_runs, GENERATIONS))
    government_cost = 0.0
    government_revenue = 0.0

    for run in range(n_runs):
        assets_next_gen = 0
        for gen in range(GENERATIONS):
            age_assets = assets_next_gen
            yearly_income_mean = DEFAULT_INCOME_MEAN

            for age in range(LIFESPAN):
                # Youth: No income, consumption only.
                if age < 22:
                    income = 0
                    consumption = DEFAULT_YOUTH_CONSUMPTION
                    tax = bracket_income_tax(income)
                    government_revenue += tax
                    net_income = income - tax
                # Working years.
                elif age < 65:
                    yearly_income_mean *= (1 + wage_growth)
                    yearly_income_mean *= (1 + productivity_growth * (age / LIFESPAN))
                    raw_income = max(0, np.random.normal(yearly_income_mean, DEFAULT_INCOME_STD))
                    if np.random.rand() < unemployment_prob:
                        income = unemployment_benefit
                    else:
                        income = raw_income
                    tax = bracket_income_tax(income)
                    government_revenue += tax
                    net_income = income - tax

                    base_rate = 0.55 if income < 60000 else 0.60
                    c_rate = max(0, base_rate - forced_saving)
                    consumption = income * c_rate
                    if np.random.rand() < overspending_prob:
                        consumption *= overspending_factor
                    if housing_choice:
                        consumption += mortgage_payment
                # Retirement.
                else:
                    income = DEFAULT_RETIREMENT_INCOME
                    consumption = DEFAULT_RETIREMENT_CONSUMPTION + old_age_medical
                    tax = bracket_income_tax(income)
                    government_revenue += tax
                    net_income = income - tax

                savings = net_income - consumption

                # UBC payment at age 21.
                if age == 21 and ubc > 0:
                    age_assets += ubc
                    government_cost += ubc
                
                # Apply wealth tax if assets exceed threshold.
                if age_assets > WEALTH_TAX_THRESHOLD:
                    tax_w = (age_assets - WEALTH_TAX_THRESHOLD) * WEALTH_TAX_RATE
                    age_assets -= tax_w
                    government_revenue += tax_w

                # Compute asset returns.
                curr_interest = base_interest + age * interest_trend
                real_return = apply_asset_allocation(
                    asset=age_assets,
                    stocks_fraction=stocks_fraction,
                    base_interest=curr_interest,
                    inflation=inflation,
                    shock_prob=shock_prob,
                    shock_impact=shock_impact
                )
                age_assets = (age_assets + savings) * (1 + real_return)

            # End-of-life inheritance.
            inherited = age_assets / n_children
            i_tax_rate = progressive_inheritance_tax(inherited)
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
    """
    Single lifecycle simulation for demonstration.
    Returns arrays for ages, income, consumption, savings, assets, and cumulative savings.
    """
    ages = np.arange(LIFESPAN)
    income_array = np.zeros(LIFESPAN)
    consumption_array = np.zeros(LIFESPAN)
    savings_array = np.zeros(LIFESPAN)
    assets_array = np.zeros(LIFESPAN)
    cumulative_savings_array = np.zeros(LIFESPAN)

    age_assets = 0
    yearly_income_mean = DEFAULT_INCOME_MEAN

    for i, age in enumerate(ages):
        if age < 22:
            income = 0
            consumption = DEFAULT_YOUTH_CONSUMPTION
            tax = bracket_income_tax(income)
            net_income = income - tax
        elif age < 65:
            yearly_income_mean *= (1 + wage_growth)
            yearly_income_mean *= (1 + productivity_growth * (age / LIFESPAN))
            raw_income = max(0, np.random.normal(yearly_income_mean, DEFAULT_INCOME_STD))
            if np.random.rand() < unemployment_prob:
                income = unemployment_benefit
            else:
                income = raw_income
            tax = bracket_income_tax(income)
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
            tax = bracket_income_tax(income)
            net_income = income - tax

        savings = net_income - consumption

        if age == 21 and ubc > 0:
            age_assets += ubc

        if age_assets > WEALTH_TAX_THRESHOLD:
            tax_w = (age_assets - WEALTH_TAX_THRESHOLD) * WEALTH_TAX_RATE
            age_assets -= tax_w

        curr_interest = base_interest + i * interest_trend
        real_return = apply_asset_allocation(
            asset=age_assets,
            stocks_fraction=stocks_fraction,
            base_interest=curr_interest,
            inflation=inflation,
            shock_prob=shock_prob,
            shock_impact=shock_impact
        )
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

##############################################################################
#               TKINTER APP: WORST ECONOMY + POSITIVE WEALTH (Min Cost)
##############################################################################

class AdvancedPolicyApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Worst Economy + Positive Wealth (Minimize Cost)")
        main_frame = ttk.Frame(self)
        main_frame.pack(pady=5, padx=5, fill="x")
        row_idx = 0

        # Basic "bad economy" parameters with range entries.
        row_idx = self.add_labeled_entry(main_frame, "Inflation", "0.05", row_idx)
        row_idx = self.add_range_entries(main_frame, "Inflation", "0.0", "0.1", row_idx)
        row_idx = self.add_labeled_entry(main_frame, "ShockProbability", "0.15", row_idx)
        row_idx = self.add_range_entries(main_frame, "ShockProbability", "0.0", "0.5", row_idx)
        row_idx = self.add_labeled_entry(main_frame, "ShockImpact", "0.3", row_idx)
        row_idx = self.add_range_entries(main_frame, "ShockImpact", "0.0", "0.5", row_idx)

        # Additional policy parameters.
        row_idx = self.add_labeled_entry(main_frame, "ForcedSaving", "0.1", row_idx)
        row_idx = self.add_labeled_entry(main_frame, "Guarantee", "10000", row_idx)
        row_idx = self.add_labeled_entry(main_frame, "UBC", "20000", row_idx)

        ttk.Label(main_frame, text="=== More Params ===").grid(row=row_idx, column=0, columnspan=2, pady=5)
        row_idx += 1
        row_idx = self.add_labeled_entry(main_frame, "ChildrenPerFamily", "2", row_idx)
        row_idx = self.add_labeled_entry(main_frame, "WageGrowth", "0.01", row_idx)
        row_idx = self.add_labeled_entry(main_frame, "UnemploymentProb", "0.05", row_idx)
        row_idx = self.add_labeled_entry(main_frame, "UnemploymentBenefit", "5000", row_idx)
        row_idx = self.add_labeled_entry(main_frame, "BaseInterest", "0.02", row_idx)
        row_idx = self.add_labeled_entry(main_frame, "InterestTrend", "0.0", row_idx)
        row_idx = self.add_labeled_entry(main_frame, "ProductivityGrowth", "0.0", row_idx)
        row_idx = self.add_labeled_entry(main_frame, "OldAgeMedical", "2000", row_idx)
        row_idx = self.add_labeled_entry(main_frame, "MortgagePayment", "5000", row_idx)
        row_idx = self.add_labeled_entry(main_frame, "StocksFraction", "0.5", row_idx)
        row_idx = self.add_labeled_entry(main_frame, "OverspendingProb", "0.05", row_idx)
        row_idx = self.add_labeled_entry(main_frame, "OverspendingFactor", "1.2", row_idx)

        self.housing_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(main_frame, text="EnableHousing", variable=self.housing_var)\
            .grid(row=row_idx, column=0, columnspan=3, sticky="w", pady=3)
        row_idx += 1

        # Buttons for simulation, random search, and GA optimization.
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=5)
        run_button = ttk.Button(btn_frame, text="Run Simulation", command=self.run_simulation)
        run_button.grid(row=0, column=0, padx=10)
        worst_button = ttk.Button(btn_frame, text="Worst Economy + Positive (Min Cost)", command=self.run_worst_economy_search)
        worst_button.grid(row=0, column=1, padx=10)
        ga_button = ttk.Button(btn_frame, text="Run GA Optimization", command=self.run_ga_optimization)
        ga_button.grid(row=0, column=2, padx=10)

        self.results_box = tk.Text(self, width=80, height=20)
        self.results_box.pack(pady=5)
        self.N_SAMPLES = 50  # for random search

    def add_labeled_entry(self, frame, label_text, default_val, row_idx):
        ttk.Label(frame, text=label_text).grid(row=row_idx, column=0, sticky="e")
        var = tk.StringVar(value=default_val)
        ttk.Entry(frame, textvariable=var, width=8).grid(row=row_idx, column=1, padx=5)
        setattr(self, label_text, var)
        return row_idx + 1

    def add_range_entries(self, frame, param, default_min, default_max, row_idx):
        ttk.Label(frame, text=f"{param}Min").grid(row=row_idx, column=2, sticky="e")
        var_min = tk.StringVar(value=default_min)
        ttk.Entry(frame, textvariable=var_min, width=6).grid(row=row_idx, column=3, padx=3)
        ttk.Label(frame, text=f"{param}Max").grid(row=row_idx, column=4, sticky="e")
        var_max = tk.StringVar(value=default_max)
        ttk.Entry(frame, textvariable=var_max, width=6).grid(row=row_idx, column=5, padx=3)
        setattr(self, f"{param}Min", var_min)
        setattr(self, f"{param}Max", var_max)
        return row_idx + 1

    def get_float(self, var):
        try:
            return float(var.get())
        except:
            return 0.0

    def get_bounds(self, param):
        var_min = getattr(self, f"{param}Min", None)
        var_max = getattr(self, f"{param}Max", None)
        lo, hi = 0.0, 1.0
        if var_min and var_max:
            try:
                lo_ = float(var_min.get())
                hi_ = float(var_max.get())
                if lo_ < hi_:
                    lo, hi = lo_, hi_
            except:
                pass
        return (lo, hi)

    def parse_params(self):
        p = {}
        p["inflation"] = self.get_float(self.Inflation)
        p["shock_prob"] = self.get_float(self.ShockProbability)
        p["shock_impact"] = self.get_float(self.ShockImpact)
        p["forced_saving"] = self.get_float(self.ForcedSaving)
        p["guarantee"] = self.get_float(self.Guarantee)
        p["ubc"] = self.get_float(self.UBC)
        p["n_children"] = int(self.get_float(self.ChildrenPerFamily))
        p["wage_growth"] = self.get_float(self.WageGrowth)
        p["unemployment_prob"] = self.get_float(self.UnemploymentProb)
        p["unemployment_benefit"] = self.get_float(self.UnemploymentBenefit)
        p["base_interest"] = self.get_float(self.BaseInterest)
        p["interest_trend"] = self.get_float(self.InterestTrend)
        p["productivity_growth"] = self.get_float(self.ProductivityGrowth)
        p["old_age_medical"] = self.get_float(self.OldAgeMedical)
        p["housing_choice"] = self.housing_var.get()
        p["mortgage_payment"] = self.get_float(self.MortgagePayment)
        p["stocks_fraction"] = self.get_float(self.StocksFraction)
        p["overspending_prob"] = self.get_float(self.OverspendingProb)
        p["overspending_factor"] = self.get_float(self.OverspendingFactor)
        return p

    def run_simulation(self):
        self.results_box.delete("1.0", tk.END)
        try:
            p = self.parse_params()
        except ValueError:
            self.results_box.insert(tk.END, "Invalid numeric input.\n")
            return

        gen_wealth, gov_cost, gov_rev = simulate_three_generations(
            inflation=p["inflation"],
            shock_prob=p["shock_prob"],
            shock_impact=p["shock_impact"],
            forced_saving=p["forced_saving"],
            guarantee=p["guarantee"],
            ubc=p["ubc"],
            n_children=p["n_children"],
            wage_growth=p["wage_growth"],
            unemployment_prob=p["unemployment_prob"],
            unemployment_benefit=p["unemployment_benefit"],
            base_interest=p["base_interest"],
            interest_trend=p["interest_trend"],
            productivity_growth=p["productivity_growth"],
            old_age_medical=p["old_age_medical"],
            housing_choice=p["housing_choice"],
            mortgage_payment=p["mortgage_payment"],
            stocks_fraction=p["stocks_fraction"],
            overspending_prob=p["overspending_prob"],
            overspending_factor=p["overspending_factor"],
            n_runs=MONTE_CARLO_RUNS
        )
        avg_wealth = gen_wealth.mean(axis=0)
        # Report average government cost and revenue per simulation run.
        avg_gov_cost = gov_cost / MONTE_CARLO_RUNS
        avg_gov_rev = gov_rev / MONTE_CARLO_RUNS
        gov_balance = avg_gov_rev - avg_gov_cost

        self.results_box.insert(tk.END, "=== 3-Gen Simulation ===\n")
        for i, val in enumerate(avg_wealth, start=1):
            self.results_box.insert(tk.END, f"Gen {i} Avg: £{val:,.2f}\n")
        self.results_box.insert(tk.END, f"\nAvg Govt Cost: £{avg_gov_cost:,.2f}\n")
        self.results_box.insert(tk.END, f"Avg Govt Revenue: £{avg_gov_rev:,.2f}\n")
        self.results_box.insert(tk.END, f"Avg Govt Balance: £{gov_balance:,.2f}\n")

        plt.figure(figsize=(7,4))
        plt.plot(range(1, GENERATIONS+1), avg_wealth, marker='o', color='orange')
        plt.title("Average Final Wealth Over 3 Generations")
        plt.xlabel("Generation")
        plt.ylabel("Wealth (£)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        (ages, inc, cons, sav, ast, cum_sav) = simulate_single_lifecycle(
            inflation=p["inflation"],
            shock_prob=p["shock_prob"],
            shock_impact=p["shock_impact"],
            forced_saving=p["forced_saving"],
            guarantee=p["guarantee"],
            ubc=p["ubc"],
            n_children=p["n_children"],
            wage_growth=p["wage_growth"],
            unemployment_prob=p["unemployment_prob"],
            unemployment_benefit=p["unemployment_benefit"],
            base_interest=p["base_interest"],
            interest_trend=p["interest_trend"],
            productivity_growth=p["productivity_growth"],
            old_age_medical=p["old_age_medical"],
            housing_choice=p["housing_choice"],
            mortgage_payment=p["mortgage_payment"],
            stocks_fraction=p["stocks_fraction"],
            overspending_prob=p["overspending_prob"],
            overspending_factor=p["overspending_factor"]
        )

        plt.figure(figsize=(10,6))
        plt.plot(ages, ast, label='Assets', color='orange')
        plt.plot(ages, cum_sav, label='Cumulative Savings', color='red')
        plt.plot(ages, inc, label='Income', color='deeppink')
        plt.plot(ages, cons, label='Consumption', color='hotpink')
        plt.axvline(x=21, color='green', linestyle='--', label='UBC@21')
        plt.axvline(x=65, color='gray', linestyle='--', label='Retire@65')
        plt.title("Single Lifecycle")
        plt.xlabel("Age")
        plt.ylabel("Wealth (£)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def run_worst_economy_search(self):
        self.results_box.delete("1.0", tk.END)
        try:
            base_params = self.parse_params()
        except ValueError:
            self.results_box.insert(tk.END, "Invalid numeric input.\n")
            return

        infl_lo, infl_hi = self.get_bounds("Inflation")
        sp_lo, sp_hi = self.get_bounds("ShockProbability")
        si_lo, si_hi = self.get_bounds("ShockImpact")

        best_index = -1e15
        best_cost = 1e15
        best_scenario = None
        best_wealth = None

        for _ in range(self.N_SAMPLES):
            trial = dict(base_params)
            trial["inflation"] = random.uniform(infl_lo, infl_hi)
            trial["shock_prob"] = random.uniform(sp_lo, sp_hi)
            trial["shock_impact"] = random.uniform(si_lo, si_hi)

            gen_wealth, gov_cost, _ = simulate_three_generations(
                inflation=trial["inflation"],
                shock_prob=trial["shock_prob"],
                shock_impact=trial["shock_impact"],
                forced_saving=trial["forced_saving"],
                guarantee=trial["guarantee"],
                ubc=trial["ubc"],
                n_children=trial["n_children"],
                wage_growth=trial["wage_growth"],
                unemployment_prob=trial["unemployment_prob"],
                unemployment_benefit=trial["unemployment_benefit"],
                base_interest=trial["base_interest"],
                interest_trend=trial["interest_trend"],
                productivity_growth=trial["productivity_growth"],
                old_age_medical=trial["old_age_medical"],
                housing_choice=trial["housing_choice"],
                mortgage_payment=trial["mortgage_payment"],
                stocks_fraction=trial["stocks_fraction"],
                overspending_prob=trial["overspending_prob"],
                overspending_factor=trial["overspending_factor"],
                n_runs=MONTE_CARLO_RUNS
            )
            avg_wealth = gen_wealth.mean(axis=0)
            if np.any(avg_wealth <= 0):
                continue
            bad_index = trial["inflation"] + trial["shock_prob"] + trial["shock_impact"]
            if bad_index > best_index:
                best_index = bad_index
                best_cost = gov_cost
                best_scenario = trial
                best_wealth = avg_wealth
            elif abs(bad_index - best_index) < 1e-9:
                if gov_cost < best_cost:
                    best_cost = gov_cost
                    best_scenario = trial
                    best_wealth = avg_wealth

        if best_scenario is None:
            self.results_box.insert(tk.END, "No scenario found with all generations > 0.\n")
            return

        self.results_box.insert(tk.END, "=== WORST ECONOMY + POSITIVE GENS (Min Cost) ===\n")
        self.results_box.insert(tk.END, f"badIndex = inflation + shockProb + shockImpact = {best_index:.4f}\n")
        self.results_box.insert(tk.END, f"Avg Govt Cost: £{best_cost/ MONTE_CARLO_RUNS:,.2f}\n\n")
        for k, v in best_scenario.items():
            self.results_box.insert(tk.END, f"{k}: {v:.4f}\n")
        self.results_box.insert(tk.END, "\nAverage Final Wealth by Generation:\n")
        for i, val in enumerate(best_wealth, start=1):
            self.results_box.insert(tk.END, f" Gen {i}: £{val:,.2f}\n")

        plt.figure(figsize=(7,4))
        plt.plot(range(1, GENERATIONS+1), best_wealth, marker='o', color='orange')
        plt.title("Worst Economy + All Gen>0, Min Cost (Avg Wealth)")
        plt.xlabel("Generation")
        plt.ylabel("Wealth (£)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        (ages, inc, cons, sav, ast, cum_sav) = simulate_single_lifecycle(
            inflation=best_scenario["inflation"],
            shock_prob=best_scenario["shock_prob"],
            shock_impact=best_scenario["shock_impact"],
            forced_saving=best_scenario["forced_saving"],
            guarantee=best_scenario["guarantee"],
            ubc=best_scenario["ubc"],
            n_children=best_scenario["n_children"],
            wage_growth=best_scenario["wage_growth"],
            unemployment_prob=best_scenario["unemployment_prob"],
            unemployment_benefit=best_scenario["unemployment_benefit"],
            base_interest=best_scenario["base_interest"],
            interest_trend=best_scenario["interest_trend"],
            productivity_growth=best_scenario["productivity_growth"],
            old_age_medical=best_scenario["old_age_medical"],
            housing_choice=best_scenario["housing_choice"],
            mortgage_payment=best_scenario["mortgage_payment"],
            stocks_fraction=best_scenario["stocks_fraction"],
            overspending_prob=best_scenario["overspending_prob"],
            overspending_factor=best_scenario["overspending_factor"]
        )

        plt.figure(figsize=(10,6))
        plt.plot(ages, ast, label='Assets', color='orange')
        plt.plot(ages, cum_sav, label='Cumulative Savings', color='red')
        plt.plot(ages, inc, label='Income', color='deeppink')
        plt.plot(ages, cons, label='Consumption', color='hotpink')
        plt.axvline(x=21, color='green', linestyle='--', label='UBC@21')
        plt.axvline(x=65, color='gray', linestyle='--', label='Retire@65')
        plt.title("Worst Economy + Positive Gens, Min Cost: Single Lifecycle")
        plt.xlabel("Age")
        plt.ylabel("Wealth (£)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def run_ga_optimization(self):
        self.results_box.delete("1.0", tk.END)
        try:
            base_params = self.parse_params()
        except ValueError:
            self.results_box.insert(tk.END, "Invalid numeric input.\n")
            return

        infl_lo, infl_hi = self.get_bounds("Inflation")
        sp_lo, sp_hi = self.get_bounds("ShockProbability")
        si_lo, si_hi = self.get_bounds("ShockImpact")

        # Define gene ranges for Guarantee and UBC based on current user input.
        guarantee_lo = self.get_float(self.Guarantee) * 0.5
        guarantee_hi = self.get_float(self.Guarantee) * 1.5
        ubc_lo = 0
        ubc_hi = self.get_float(self.UBC) * 1.5

        n_runs_ga = 200  # Fewer runs for faster GA evaluation.

        # Verbose callback to report GA progress.
        def callback_generation(ga_instance):
            generation = ga_instance.generations_completed
            best_solution, best_fitness, _ = ga_instance.best_solution()
            print(f"Generation {generation}: Best Fitness = {best_fitness:.4f}")

        # Fitness function: maximize extreme worst-case parameters while ensuring viability and a positive (or non-negative)
        # government balance, using average government cost and revenue.
        def fitness_func(ga_instance, solution, solution_idx):
            inflation_ga, shock_prob_ga, shock_impact_ga, guarantee_ga, ubc_ga = solution
            trial = dict(base_params)
            trial["inflation"] = inflation_ga
            trial["shock_prob"] = shock_prob_ga
            trial["shock_impact"] = shock_impact_ga
            trial["guarantee"] = guarantee_ga
            trial["ubc"] = ubc_ga

            gen_wealth, gov_cost, gov_rev = simulate_three_generations(
                inflation=trial["inflation"],
                shock_prob=trial["shock_prob"],
                shock_impact=trial["shock_impact"],
                forced_saving=trial["forced_saving"],
                guarantee=trial["guarantee"],
                ubc=trial["ubc"],
                n_children=trial["n_children"],
                wage_growth=trial["wage_growth"],
                unemployment_prob=trial["unemployment_prob"],
                unemployment_benefit=trial["unemployment_benefit"],
                base_interest=trial["base_interest"],
                interest_trend=trial["interest_trend"],
                productivity_growth=trial["productivity_growth"],
                old_age_medical=trial["old_age_medical"],
                housing_choice=trial["housing_choice"],
                mortgage_payment=trial["mortgage_payment"],
                stocks_fraction=trial["stocks_fraction"],
                overspending_prob=trial["overspending_prob"],
                overspending_factor=trial["overspending_factor"],
                n_runs=n_runs_ga
            )
            avg_wealth = gen_wealth.mean(axis=0)
            if np.any(avg_wealth <= 0):
                return -1e6

            # Compute average government cost and revenue per run.
            avg_gov_cost = gov_cost / n_runs_ga
            avg_gov_rev = gov_rev / n_runs_ga
            government_balance = avg_gov_rev - avg_gov_cost
            # If government balance is negative, penalize heavily.
            if government_balance < 0:
                return -1e6

            bad_index = inflation_ga + shock_prob_ga + shock_impact_ga
            min_wealth = avg_wealth.min()
            # Compose fitness: maximize bad_index, strongly penalize high average government cost and high minimum wealth,
            # and add bonus for a higher government balance.
            fitness = bad_index - (avg_gov_cost / 5e5) - (min_wealth / 1e7) + (government_balance / 1e5)
            return fitness

        ga_instance = pygad.GA(
            num_generations=50,
            num_parents_mating=10,
            fitness_func=fitness_func,
            sol_per_pop=20,
            num_genes=5,
            gene_space=[
                {"low": infl_lo, "high": infl_hi},
                {"low": sp_lo, "high": sp_hi},
                {"low": si_lo, "high": si_hi},
                {"low": guarantee_lo, "high": guarantee_hi},
                {"low": ubc_lo, "high": ubc_hi}
            ],
            mutation_percent_genes=20,
            mutation_num_genes=1,
            mutation_type="random",
            crossover_type="single_point",
            stop_criteria="saturate_10",
            on_generation=callback_generation
        )

        self.results_box.insert(tk.END, "Running GA Optimization...\n")
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        best_inflation, best_shock_prob, best_shock_impact, best_guarantee, best_ubc = solution
        gen_wealth, gov_cost, gov_rev = simulate_three_generations(
            inflation=best_inflation,
            shock_prob=best_shock_prob,
            shock_impact=best_shock_impact,
            forced_saving=base_params["forced_saving"],
            guarantee=best_guarantee,
            ubc=best_ubc,
            n_children=base_params["n_children"],
            wage_growth=base_params["wage_growth"],
            unemployment_prob=base_params["unemployment_prob"],
            unemployment_benefit=base_params["unemployment_benefit"],
            base_interest=base_params["base_interest"],
            interest_trend=base_params["interest_trend"],
            productivity_growth=base_params["productivity_growth"],
            old_age_medical=base_params["old_age_medical"],
            housing_choice=base_params["housing_choice"],
            mortgage_payment=base_params["mortgage_payment"],
            stocks_fraction=base_params["stocks_fraction"],
            overspending_prob=base_params["overspending_prob"],
            overspending_factor=base_params["overspending_factor"],
            n_runs=MONTE_CARLO_RUNS
        )
        avg_wealth = gen_wealth.mean(axis=0)
        avg_gov_cost = gov_cost / MONTE_CARLO_RUNS
        avg_gov_rev = gov_rev / MONTE_CARLO_RUNS
        government_balance = avg_gov_rev - avg_gov_cost
        bad_index = best_inflation + best_shock_prob + best_shock_impact

        self.results_box.insert(tk.END, "=== GA Optimization Result ===\n")
        self.results_box.insert(tk.END, f"Best Fitness: {solution_fitness:.4f}\n")
        self.results_box.insert(tk.END, f"badIndex: {bad_index:.4f}\n")
        self.results_box.insert(tk.END, f"Avg Govt Cost: £{avg_gov_cost:,.2f}\n")
        self.results_box.insert(tk.END, f"Avg Govt Revenue: £{avg_gov_rev:,.2f}\n")
        self.results_box.insert(tk.END, f"Avg Govt Balance: £{government_balance:,.2f}\n")
        self.results_box.insert(tk.END, f"Best Inflation: {best_inflation:.4f}\n")
        self.results_box.insert(tk.END, f"Best ShockProbability: {best_shock_prob:.4f}\n")
        self.results_box.insert(tk.END, f"Best ShockImpact: {best_shock_impact:.4f}\n")
        self.results_box.insert(tk.END, f"Best Guarantee: {best_guarantee:.4f}\n")
        self.results_box.insert(tk.END, f"Best UBC: {best_ubc:.4f}\n")
        self.results_box.insert(tk.END, "\nAverage Final Wealth by Generation:\n")
        for i, val in enumerate(avg_wealth, start=1):
            self.results_box.insert(tk.END, f" Gen {i}: £{val:,.2f}\n")

        plt.figure(figsize=(7,4))
        plt.plot(range(1, GENERATIONS+1), avg_wealth, marker='o', color='orange')
        plt.title("GA-Optimized 3-Gen Wealth")
        plt.xlabel("Generation")
        plt.ylabel("Wealth (£)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        (ages, inc, cons, sav, ast, cum_sav) = simulate_single_lifecycle(
            inflation=best_inflation,
            shock_prob=best_shock_prob,
            shock_impact=best_shock_impact,
            forced_saving=base_params["forced_saving"],
            guarantee=best_guarantee,
            ubc=best_ubc,
            n_children=base_params["n_children"],
            wage_growth=base_params["wage_growth"],
            unemployment_prob=base_params["unemployment_prob"],
            unemployment_benefit=base_params["unemployment_benefit"],
            base_interest=base_params["base_interest"],
            interest_trend=base_params["interest_trend"],
            productivity_growth=base_params["productivity_growth"],
            old_age_medical=base_params["old_age_medical"],
            housing_choice=base_params["housing_choice"],
            mortgage_payment=base_params["mortgage_payment"],
            stocks_fraction=base_params["stocks_fraction"],
            overspending_prob=base_params["overspending_prob"],
            overspending_factor=base_params["overspending_factor"]
        )

        plt.figure(figsize=(10,6))
        plt.plot(ages, ast, label='Assets', color='orange')
        plt.plot(ages, cum_sav, label='Cumulative Savings', color='red')
        plt.plot(ages, inc, label='Income', color='deeppink')
        plt.plot(ages, cons, label='Consumption', color='hotpink')
        plt.axvline(x=21, color='green', linestyle='--', label='UBC@21')
        plt.axvline(x=65, color='gray', linestyle='--', label='Retire@65')
        plt.title("GA Optimization: Single Lifecycle")
        plt.xlabel("Age")
        plt.ylabel("Wealth (£)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def main():
    app = AdvancedPolicyApp()
    app.mainloop()

if __name__ == "__main__":
    main()
