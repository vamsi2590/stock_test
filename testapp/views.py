from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
import os
from django.conf import settings
import matplotlib.pyplot as plt
import io
import urllib
import base64
from mftool import Mftool
import json
from .forms import CAGRForm
from .utils import calculate_cagr
import random
import math


# Initialize Mftool
mf = Mftool()

# Sign Up View
def SignupPage(request):
    if request.method == 'POST':
        uname = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('password1')
        pass2 = request.POST.get('password2')

        if pass1 != pass2:
            return HttpResponse("Your password and confirm password do not match!")

        if User.objects.filter(username=uname).exists():
            return HttpResponse("Username already exists!")

        User.objects.create_user(username=uname, email=email, password=pass1)
        return redirect('login')

    return render(request, 'mutual_funds/signup.html')

# Login View
def LoginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            return HttpResponse("Username or Password is incorrect!")

    return render(request, 'mutual_funds/login.html')

# Logout View
def LogoutPage(request):
    logout(request)
    return redirect('login')

# Home Page
@login_required(login_url='login')
def HomePage(request):
    return render(request, 'mutual_funds/home.html')

# Fetch Scheme Names
def get_scheme_names():
    try:
        scheme_mapping = mf.get_scheme_codes()
        return {v: k for k, v in scheme_mapping.items()} if scheme_mapping else {}
    except Exception:
        return {}

# Dashboard View
@login_required(login_url='login')
def dashboard(request):
    scheme_names = get_scheme_names()
    return render(request, 'mutual_funds/dashboard.html', {'scheme_names': scheme_names})

# Scheme Details View
@login_required(login_url='login')
def scheme_details(request):
    scheme_names = get_scheme_names()
    selected_scheme = request.GET.get('scheme')
    scheme_details = None

    if selected_scheme and selected_scheme in scheme_names:
        scheme_code = scheme_names[selected_scheme]
        scheme_details = mf.get_scheme_details(scheme_code)

    return render(request, 'mutual_funds/scheme_details.html', {
        'scheme_names': scheme_names,
        'selected_scheme': selected_scheme,
        'scheme_details': scheme_details
    })

# Historical NAV View
@login_required(login_url='login')
def historical_nav(request):
    scheme_code = request.GET.get('scheme_code', '')
    schemes = mf.get_scheme_codes()
    nav_data = mf.get_scheme_historical_nav(scheme_code) if scheme_code else None

    return render(request, 'mutual_funds/historical_nav.html', {
        'schemes': schemes,
        'nav_data': nav_data,
        'selected_scheme': scheme_code
    })

# Compare NAVs View
@login_required(login_url='login')


def compare_navs(request):
    scheme_names = mf.get_scheme_codes()
    selected_schemes = request.GET.getlist('schemes')

    if not selected_schemes:
        return render(request, 'mutual_funds/compare_navs.html', {
            'scheme_names': scheme_names,
            'selected_schemes': [],
            'error': "⚠️ No schemes selected. Please choose at least one."
        })

    comparison_df = pd.DataFrame()
    missing_schemes = []

    for scheme in selected_schemes:
        scheme_code = scheme_names.get(scheme)
        if scheme_code:
            try:
                print(f"🔍 Fetching NAV data for scheme: {scheme} (Code: {scheme_code})")
                data = mf.get_scheme_historical_nav(scheme_code, as_Dataframe=True)

                if data is None or data.empty:
                    print(f"⚠️ No NAV data for {scheme}")
                    missing_schemes.append(scheme)
                    continue  # Skip if no data

                # Fix date parsing
                data.index = pd.to_datetime(data.index, format="%d-%m-%Y", errors="coerce")
                data = data.sort_index()
                data['nav'] = data['nav'].replace(0, np.nan).interpolate()

                comparison_df[scheme] = data['nav']
            except Exception as e:
                print(f"❌ Error fetching data for {scheme}: {e}")

    if not comparison_df.empty:
        print("✅ Comparison DataFrame:")
        print(comparison_df.head())  # Debugging log

        fig = px.line(comparison_df, title="Comparison of NAVs", labels={"value": "NAV", "variable": "Scheme"})
        chart_json = json.dumps(fig, cls=px.utils.PlotlyJSONEncoder)

        print(f"✅ Chart JSON generated: {chart_json[:500]}...")  # Print first 500 chars for debugging

        return render(request, 'mutual_funds/compare_navs.html', {
            'scheme_names': scheme_names,
            'selected_schemes': selected_schemes,
            'chart_json': chart_json
        })

    error_message = "⚠️ No NAV data available for the selected schemes."
    if missing_schemes:
        error_message += f" These schemes might not have historical NAV data: {', '.join(missing_schemes)}."

    print("❌ No valid NAV data found.")
    return render(request, 'mutual_funds/compare_navs.html', {
        'scheme_names': scheme_names,
        'selected_schemes': selected_schemes,
        'error': error_message
    })
@login_required(login_url='login')
def performance_heatmap(request):
    """Fetch and process NAV data for heatmap"""
    try:
        all_schemes = mf.get_scheme_codes()  # Fetch real-time scheme codes

        # Convert dictionary to a list of dictionaries for template rendering
        scheme_list = [{'code': code, 'name': name} for code, name in all_schemes.items()]
    except Exception:
        scheme_list = []  # Default to an empty list in case of an error

    heatmap_url = None
    error = None
    selected_scheme_code = request.GET.get('scheme')  # Get selected scheme from request
    selected_scheme_name = None

    if selected_scheme_code:
        try:
            # Get the scheme name using the selected scheme code
            selected_scheme_name = next((scheme['name'] for scheme in scheme_list if scheme['code'] == selected_scheme_code), None)

            # Fetch historical NAV data for the selected scheme
            nav_data = mf.get_scheme_historical_nav(selected_scheme_code, as_Dataframe=True)

            if not isinstance(nav_data, pd.DataFrame) or nav_data.empty:
                raise ValueError("No historical NAV data available.")

            # Process the data to match the structure
            nav_data = nav_data.reset_index().rename(columns={'index': 'date'})
            nav_data['date'] = pd.to_datetime(nav_data['date'], format='%d-%m-%Y', errors='coerce')

            # Drop invalid date entries
            nav_data = nav_data.dropna(subset=['date'])

            # Extract the month from the date
            nav_data['month'] = nav_data['date'].dt.month
            nav_data['nav'] = pd.to_numeric(nav_data['nav'], errors='coerce')  # Convert NAV to float

            # Drop missing NAV values
            nav_data = nav_data.dropna(subset=['nav'])

            # Group by month and calculate the mean NAV for that month
            heatmap_data = nav_data.groupby(['month'])['nav'].mean().reset_index()

            # Create heatmap using Plotly
            fig = px.density_heatmap(heatmap_data, x='month', y='nav', title="NAV Performance Heatmap", color_continuous_scale="Viridis")

            # Explicitly set the x-axis labels for months (1-12)
            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Months
                    ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']  # Month names
                ),
                yaxis_title='Average NAV',
                xaxis_title='Month'
            )

            # Ensure media directory exists
            media_dir = os.path.join(settings.MEDIA_ROOT, 'heatmaps')
            os.makedirs(media_dir, exist_ok=True)

            # Save heatmap image
            heatmap_path = os.path.join(media_dir, 'heatmap.png')
            fig.write_image(heatmap_path)

            # Provide correct media URL
            heatmap_url = settings.MEDIA_URL + 'heatmaps/heatmap.png'

        except Exception as e:
            error = f"An error occurred while fetching data for the selected scheme: {str(e)}"

    return render(request, 'mutual_funds/performance_heatmap.html', {
        'scheme_list': scheme_list,
        'heatmap_url': heatmap_url,
        'error': error,
        'selected_scheme_name': selected_scheme_name  # Pass selected scheme name to the template
    })

@login_required(login_url='login')
def average_aum(request):
    try:
        # Fetch Average AUM Data
        aum_data = mf.get_average_aum('July - September 2024', False)

        if aum_data and isinstance(aum_data, list):
            aum_df = pd.DataFrame(aum_data)

            # Check if required columns exist
            if {'AAUM Overseas', 'AAUM Domestic', 'Fund Name'}.issubset(aum_df.columns):
                # Convert AUM columns to numeric
                aum_df[['AAUM Overseas', 'AAUM Domestic']] = aum_df[['AAUM Overseas', 'AAUM Domestic']].apply(pd.to_numeric, errors='coerce')

                # Calculate total AUM
                aum_df['Total AUM'] = aum_df[['AAUM Overseas', 'AAUM Domestic']].sum(axis=1)

                # Rename columns properly
                aum_df.rename(columns={'Fund Name': 'fund_name', 'Total AUM': 'total_aum'}, inplace=True)

                # Convert DataFrame to list of dictionaries for template rendering
                aum_list = aum_df[['fund_name', 'total_aum']].to_dict(orient='records')
            else:
                aum_list = []
        else:
            aum_list = []

    except Exception as e:
        aum_list = []
        logging.error(f"Error fetching AUM data: {e}")

    return render(request, 'mutual_funds/average_aum.html', {'aum_list': aum_list})





def risk_volatility_analysis(request):
    """Analyze risk and volatility of a mutual fund scheme."""
    scheme_names = mf.get_scheme_codes()
    selected_scheme_code = request.GET.get('scheme')
    selected_scheme_name = scheme_names.get(selected_scheme_code, None)
    metrics = {}
    error = None

    if selected_scheme_code:
        try:
            # Fetch NAV data
            nav_data = mf.get_scheme_historical_nav(selected_scheme_code, as_Dataframe=True)

            if nav_data.empty:
                error = "No valid NAV data returned for this scheme."
            else:
                # Process NAV data
                nav_data = nav_data.reset_index().rename(columns={'index': 'date'})
                nav_data['date'] = pd.to_datetime(nav_data['date'], dayfirst=True)
                nav_data['nav'] = pd.to_numeric(nav_data['nav'], errors='coerce')
                nav_data = nav_data.dropna(subset=['nav'])  # Remove NaN values

                if nav_data.empty:
                    error = "No valid NAV data available after filtering."

                else:
                    # Calculate risk metrics
                    nav_data['returns'] = nav_data['nav'].pct_change()
                    nav_data = nav_data.dropna(subset=['returns'])  # Remove NaN returns

                    annualized_volatility = nav_data['returns'].std() * np.sqrt(252)
                    annualized_return = (1 + nav_data['returns'].mean()) ** 252 - 1
                    risk_free_rate = 0.06
                    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

                    # Store results
                    metrics = {
                        "Annualized Volatility": f"{annualized_volatility:.2%}",
                        "Annualized Return": f"{annualized_return:.2%}",
                        "Sharpe Ratio": f"{sharpe_ratio:.2f}"
                    }

        except Exception as e:
            error = f"An error occurred while fetching data: {str(e)}"

    return render(request, 'mutual_funds/risk_volatility.html', {
        'scheme_names': scheme_names,
        'selected_scheme_code': selected_scheme_code,
        'selected_scheme_name': selected_scheme_name,
        'metrics': metrics,
        'error': error
    })

def cagr_calculator_view(request):
    result = None

    if request.method == "POST":
        form = CAGRForm(request.POST)
        if form.is_valid():
            initial_value = form.cleaned_data['initial_value']
            final_value = form.cleaned_data['final_value']
            years = form.cleaned_data['years']
            result = calculate_cagr(initial_value, final_value, years)
        else:
            # If form is invalid, keep the submitted values
            initial_value = request.POST.get('initial_value', 1000)
            final_value = request.POST.get('final_value', 10000)
            years = request.POST.get('years', 10)
    else:
        # Generate random values only on first page load (GET request)
        initial_value = random.randint(1000, 10000)
        final_value = random.randint(10000, 100000)
        years = random.randint(1, 30)

    context = {
        'result': result,
        'initial_value': initial_value,
        'final_value': final_value,
        'years': years,
    }
    return render(request, 'mutual_funds/cagr_calculator.html', context)
def calculators_list(request):
    return render(request, 'mutual_funds/calculators.html')
def roi_calculator_view(request):
    result = None
    initial_value = 1000
    final_value = 5000

    if request.method == "POST":
        initial_value = float(request.POST.get("initial_value", 1000))
        final_value = float(request.POST.get("final_value", 5000))

        # ROI Calculation
        if initial_value > 0:
            result = ((final_value - initial_value) / initial_value) * 100
            result = round(result, 2)

    return render(request, "mutual_funds/roi_calculator.html", {
        "result": result,
        "initial_value": initial_value,
        "final_value": final_value,
    })
def stock_profit_loss_calculator(request):
    # Default values
    default_buy_price = 100
    default_sell_price = 200
    default_shares = 10

    # Values that will be retained after form submission
    buy_price = default_buy_price
    sell_price = default_sell_price
    shares = default_shares
    result = None
    abs_result = None  # Absolute value for loss display
    percentage = None

    if request.method == "POST":
        try:
            buy_price = float(request.POST.get("buy_price", default_buy_price))
            sell_price = float(request.POST.get("sell_price", default_sell_price))
            shares = int(request.POST.get("shares", default_shares))

            if buy_price > 0 and shares > 0:
                total_cost = buy_price * shares
                total_sell = sell_price * shares
                result = total_sell - total_cost
                abs_result = abs(result)  # Compute absolute value to avoid template issues
                percentage = (result / total_cost) * 100 if total_cost > 0 else 0

        except ValueError:
            result, abs_result, percentage = None, None, None

    return render(request, "mutual_funds/stock_calculator.html", {
        "buy_price": buy_price,
        "sell_price": sell_price,
        "shares": shares,
        "result": result,
        "abs_result": abs_result,
        "percentage": percentage
    })


def sip_calculator(request):
    future_value = None
    monthly_investment = 5000  # Default value
    annual_rate = 12  # Default value
    years = 10  # Default value

    if request.method == "POST":
        monthly_investment = float(request.POST.get("monthly_investment", 0))
        annual_rate = float(request.POST.get("annual_rate", 0))
        years = int(request.POST.get("years", 0))

        # SIP formula
        r = (annual_rate / 100) / 12  # Convert annual rate to monthly
        n = years * 12  # Total months

        if r > 0:
            future_value = monthly_investment * (((1 + r) ** n - 1) / r) * (1 + r)
        else:
            future_value = monthly_investment * n  # Simple sum if rate is 0

        future_value = round(future_value, 2)

    return render(request, "mutual_funds/sip_calculator.html", {
        "future_value": future_value,
        "monthly_investment": monthly_investment,
        "annual_rate": annual_rate,
        "years": years
    })



def sip_annual_increase(request):
    # Default values
    monthly_investment = 5000
    annual_rate = 8.0
    years = 20
    annual_increase = 10.0
    future_value = None
    sip_growth = []

    if request.method == "POST":
        # Get user inputs
        monthly_investment = float(request.POST.get("monthly_investment", 5000))
        annual_rate = float(request.POST.get("annual_rate", 8.0))
        years = int(request.POST.get("years", 20))
        annual_increase = float(request.POST.get("annual_increase", 10.0))

        # Convert annual rate to monthly
        r = (annual_rate / 100) / 12
        future_value = 0
        sip_growth = [0]  # Start with zero for better graph visualization
        current_sip = monthly_investment

        for year in range(1, years + 1):
            for _ in range(12):  # Each year has 12 months
                future_value = future_value * (1 + r) + current_sip
            sip_growth.append(future_value)
            current_sip *= (1 + annual_increase / 100)  # Increase SIP annually

    return render(request, "mutual_funds/sip_annual_increase.html", {
        "monthly_investment": monthly_investment,
        "annual_rate": annual_rate,
        "years": years,
        "annual_increase": annual_increase,
        "future_value": round(future_value, 2) if future_value else None,
        "sip_growth": sip_growth,
    })


def ppf_calculator(request):
    future_value = None
    years = 15  # Default PPF duration
    investment = 5000  # Default Monthly Investment
    rate = 7.1  # Default Annual Interest Rate

    if request.method == "POST":
        investment = float(request.POST.get("investment", 5000))
        rate = float(request.POST.get("rate", 7.1))
        years = int(request.POST.get("years", 15))

        # Convert annual rate to monthly rate
        r = rate / 100 / 12
        n = years * 12

        # Calculate PPF Future Value with Monthly Compounding
        future_value = investment * (((1 + r) ** n - 1) / r) * (1 + r)

        # Generate Data for Graph
        ppf_values = []
        total_value = 0
        for i in range(1, years + 1):
            total_value = investment * (((1 + r) ** (i * 12) - 1) / r) * (1 + r)
            ppf_values.append(round(total_value, 2))

        ppf_values_json = json.dumps(ppf_values)

    return render(request, "mutual_funds/ppf_calculator.html", {
        "investment": investment,
        "rate": rate,
        "years": years,
        "future_value": round(future_value, 2) if future_value else None,
        "ppf_values": ppf_values_json if future_value else None
    })
def retirement_calculator(request):
    context = {
        "current_age": 30,
        "retirement_age": 60,
        "life_expectancy": 85,
        "current_savings": 500000,
        "monthly_expenses": 40000,
        "inflation_rate": 6,
        "expected_returns": 8,
        "pension_income": 10000,
        "employer_contributions": 5000,
        "future_corpus": None,
        "expense_projection": [],
        "savings_projection": []
    }

    if request.method == "POST":
        current_age = int(request.POST.get("current_age", 30))
        retirement_age = int(request.POST.get("retirement_age", 60))
        life_expectancy = int(request.POST.get("life_expectancy", 85))
        current_savings = float(request.POST.get("current_savings", 500000))
        monthly_expenses = float(request.POST.get("monthly_expenses", 40000))
        inflation_rate = float(request.POST.get("inflation_rate", 6)) / 100
        expected_returns = float(request.POST.get("expected_returns", 8)) / 100
        pension_income = float(request.POST.get("pension_income", 10000))
        employer_contributions = float(request.POST.get("employer_contributions", 5000))

        working_years = retirement_age - current_age
        retirement_years = life_expectancy - retirement_age

        # Calculate future monthly expenses at retirement (adjusted for inflation)
        future_monthly_expenses = monthly_expenses * ((1 + inflation_rate) ** working_years)
        future_annual_expenses = future_monthly_expenses * 12

        # Reduce expenses by pension income
        net_annual_expenses = max(future_annual_expenses - (pension_income * 12), 0)

        # Calculate required corpus at retirement
        retirement_corpus = net_annual_expenses * ((1 - (1 / (1 + expected_returns) ** retirement_years)) / expected_returns)

        # Savings Growth Calculation
        savings = current_savings
        savings_projection = []
        expense_projection = []

        for year in range(working_years + retirement_years):
            if year < working_years:
                savings += (employer_contributions * 12)
                savings *= (1 + expected_returns)
            else:
                savings -= net_annual_expenses
                net_annual_expenses *= (1 + inflation_rate)  # Expenses increase due to inflation

            savings_projection.append(savings)
            expense_projection.append(net_annual_expenses)

        context.update({
            "current_age": current_age,
            "retirement_age": retirement_age,
            "life_expectancy": life_expectancy,
            "current_savings": current_savings,
            "monthly_expenses": monthly_expenses,
            "inflation_rate": inflation_rate * 100,
            "expected_returns": expected_returns * 100,
            "pension_income": pension_income,
            "employer_contributions": employer_contributions,
            "future_corpus": round(retirement_corpus, 2),
            "expense_projection": expense_projection,
            "savings_projection": savings_projection,
        })

    return render(request, "mutual_funds/retirement_calculator.html", context)


def home_loan_emi_calculator(request):
    emi = None
    total_payment = None
    total_interest = None
    emi_chart = []

    if request.method == "POST":
        loan_amount = float(request.POST.get("loan_amount", 0))
        annual_rate = float(request.POST.get("annual_rate", 0))
        tenure_years = int(request.POST.get("tenure_years", 0))

        monthly_rate = (annual_rate / 100) / 12
        tenure_months = tenure_years * 12

        if monthly_rate > 0:
            emi = (loan_amount * monthly_rate * (1 + monthly_rate) ** tenure_months) / ((1 + monthly_rate) ** tenure_months - 1)
        else:
            emi = loan_amount / tenure_months  # In case of 0% interest

        total_payment = emi * tenure_months
        total_interest = total_payment - loan_amount

        # Generate data for EMI breakdown chart
        remaining_balance = loan_amount
        for _ in range(tenure_months):
            interest = remaining_balance * monthly_rate
            principal = emi - interest
            remaining_balance -= principal
            emi_chart.append(total_payment - remaining_balance)

    return render(request, "mutual_funds/home_loan_emi_calculator.html", {
        "emi": round(emi, 2) if emi else None,
        "total_payment": round(total_payment, 2) if total_payment else None,
        "total_interest": round(total_interest, 2) if total_interest else None,
        "loan_amount": loan_amount if request.method == "POST" else 5000000,
        "annual_rate": annual_rate if request.method == "POST" else 7.5,
        "tenure_years": tenure_years if request.method == "POST" else 20,
        "emi_chart": emi_chart
    })

def compounding_calculator(request):
    future_value = None
    principal = 100000  # Default Principal Amount
    rate = 5.0  # Default Annual Interest Rate (%)
    years = 10  # Default Investment Duration (years)
    times_compounded = 12  # Default Compounding Frequency (monthly)

    if request.method == "POST":
        principal = float(request.POST.get("principal", 0))
        rate = float(request.POST.get("rate", 0))
        years = int(request.POST.get("years", 0))
        times_compounded = int(request.POST.get("times_compounded", 0))

        # Compound Interest Formula: A = P (1 + r/n)^(nt)
        future_value = principal * math.pow((1 + (rate / 100) / times_compounded), (times_compounded * years))

    return render(request, "mutual_funds/compounding_calculator.html", {
        "principal": principal,
        "rate": rate,
        "years": years,
        "times_compounded": times_compounded,
        "future_value": round(future_value, 2) if future_value else None
    })
def fd_calculator(request):
    future_value = None
    years_list = []
    y_values = []

    if request.method == "POST":
        principal = float(request.POST.get("principal", 0))
        rate = float(request.POST.get("rate", 0)) / 100
        years = int(request.POST.get("years", 0))
        compounding = int(request.POST.get("compounding", 1))  # Monthly, Quarterly, Yearly, etc.

        # FD Future Value Calculation
        future_value = principal * ((1 + (rate / compounding)) ** (compounding * years))

        # Generating graph data
        for i in range(1, years + 1):
            years_list.append(i)
            y_values.append(principal * ((1 + (rate / compounding)) ** (compounding * i)))

    return render(request, "mutual_funds/fd_calculator.html", {
        "future_value": round(future_value, 2) if future_value else None,
        "years": years_list,
        "y_values": y_values
    })


def lumpsum_calculator(request):
    future_value = None
    principal = 100000  # Default Investment Amount
    rate_of_return = 8  # Default Annual Return %
    years = 10  # Default Investment Duration

    x_values = []  # Years
    y_values = []  # Future Investment Value

    if request.method == "POST":
        principal = float(request.POST.get("principal", 100000))
        rate_of_return = float(request.POST.get("rate_of_return", 8))
        years = int(request.POST.get("years", 10))

        # Calculate Future Value
        r = rate_of_return / 100
        future_value = principal * ((1 + r) ** years)

        # Generate graph data
        x_values = list(range(years + 1))
        y_values = [principal * ((1 + r) ** i) for i in x_values]

    return render(request, "mutual_funds/lumpsum_calculator.html", {
        "future_value": round(future_value, 2) if future_value else None,
        "principal": principal,
        "rate_of_return": rate_of_return,
        "years": years,
        "x_values": x_values if future_value else [],
        "y_values": y_values if future_value else [],
    })


def tax_saving_calculator(request):
    tax_saved = None
    income = 1000000  # Default Annual Income
    deductions = 150000  # Default Deductions (80C, 80D, etc.)
    tax_rate = 30  # Default Tax Slab %

    x_values = []  # Taxable Income
    y_values = []  # Tax Amount

    if request.method == "POST":
        income = float(request.POST.get("income", 1000000))
        deductions = float(request.POST.get("deductions", 150000))
        tax_rate = float(request.POST.get("tax_rate", 30))

        # Calculate Taxable Income
        taxable_income = max(0, income - deductions)

        # Calculate Tax Amount
        tax_saved = (income * (tax_rate / 100)) - (taxable_income * (tax_rate / 100))

        # Generate graph data
        x_values = [income - (i * 10000) for i in range(11)]  # Decreasing income with tax savings
        y_values = [(max(0, i - deductions) * (tax_rate / 100)) for i in x_values]  # Tax calculations

    return render(request, "mutual_funds/tax_saving_calculator.html", {
        "tax_saved": round(tax_saved, 2) if tax_saved else None,
        "income": income,
        "deductions": deductions,
        "tax_rate": tax_rate,
        "x_values": x_values if tax_saved else [],
        "y_values": y_values if tax_saved else [],
    })
