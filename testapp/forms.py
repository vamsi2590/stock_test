from django import forms

class CAGRForm(forms.Form):
    initial_value = forms.FloatField(label="Initial Investment", min_value=0.01)
    final_value = forms.FloatField(label="Final Investment", min_value=0.01)
    years = forms.FloatField(label="Number of Years", min_value=0.01)
