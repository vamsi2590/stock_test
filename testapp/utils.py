def calculate_cagr(initial_value, final_value, years):
    try:
        if initial_value <= 0 or final_value <= 0 or years <= 0:
            return "Invalid Input"
        cagr = ((final_value / initial_value) ** (1 / years)) - 1
        return round(cagr * 100, 2)  # Return CAGR as a percentage
    except Exception as e:
        return f"Error: {str(e)}"
