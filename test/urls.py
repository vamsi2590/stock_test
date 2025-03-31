"""
URL configuration for test project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from testapp import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('',views.HomePage, name="home"),
    path('calculators/',views.calculators_list, name='calculators_list'),
    path('calculators/cagr/',views.cagr_calculator_view, name='cagr_calculator'),
    path('calculators/roi/', views.roi_calculator_view, name='roi_calculator'),
    path("stock-calculator/",views.stock_profit_loss_calculator, name="stock_calculator"),
    path("sip-calculator/", views.sip_calculator, name="sip_calculator"),
    path('sip-annual-increase/', views.sip_annual_increase, name='sip_annual_increase'),
    path("ppf-calculator/",views.ppf_calculator, name="ppf_calculator"),
    path('retirement-calculator/',views.retirement_calculator, name='retirement_calculator'),
    path("home-loan-emi/", views.home_loan_emi_calculator, name="home_loan_emi_calculator"),
    path("compounding-calculator/",views.compounding_calculator, name="compounding_calculator"),
    path('lumpsum-calculator/', views.lumpsum_calculator, name='lumpsum_calculator'),
    path("fd-calculator/", views.fd_calculator, name="fd_calculator"),
    path('tax-saving-calculator/', views.tax_saving_calculator, name='tax_saving_calculator'),
    path('signup/', views.SignupPage, name='signup'),
    path('login/', views.LoginPage, name='login'),
    path('logout/', views.LogoutPage, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('scheme-details/', views.scheme_details, name='scheme_details'),
    path('historical-nav/', views.historical_nav, name='historical_nav'),
    path('compare-navs/', views.compare_navs, name='compare_navs'),
    path('average_aum/', views.average_aum, name='average_aum'),
    path('performance_heatmap/', views.performance_heatmap, name='performance_heatmap'),
    path('risk-volatility_analysis/', views.risk_volatility_analysis, name='risk_volatility_analysis'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
