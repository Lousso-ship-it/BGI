from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
from datetime import datetime
import json

app = FastAPI(title="Financial Data API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class EconomicIndicator(BaseModel):
    id: str
    title: str
    description: str
    source: str
    category: List[str]
    country: Optional[str] = None
    value: str
    lastUpdate: str
    frequency: str
    year: Optional[str] = None
    unit: Optional[str] = None

class FinancialData(BaseModel):
    symbol: str
    name: Optional[str] = None
    price: float
    change: Optional[float] = None
    changePercent: Optional[float] = None
    volume: Optional[float] = None
    marketCap: Optional[float] = None
    pe: Optional[float] = None
    eps: Optional[float] = None

class CompanyInfo(BaseModel):
    ticker: str
    company_name: Optional[str] = None
    long_business_summary: Optional[str] = None
    website: Optional[str] = None
    industry: Optional[str] = None
    sector: Optional[str] = None
    country: Optional[str] = None
    city: Optional[str] = None
    employees: Optional[int] = None

class ChartData(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float

# Global data storage
imf_data = None
wb_data = None
market_data = None
company_info = None

def load_csv_data():
    """Load all CSV data on startup"""
    global imf_data, wb_data, market_data, company_info
    
    try:
        # Load IMF indicators
        if os.path.exists("imf_indicators.csv"):
            imf_data = pd.read_csv("imf_indicators.csv")
            print(f"Loaded IMF data: {len(imf_data)} records")
        
        # Load World Bank indicators
        if os.path.exists("wb_indicators.csv"):
            wb_data = pd.read_csv("wb_indicators.csv")
            print(f"Loaded World Bank data: {len(wb_data)} records")
        
        # Load market data
        market_files = [f for f in os.listdir('.') if f.startswith('market_data_') and f.endswith('.csv')]
        if market_files:
            market_data = pd.read_csv(market_files[0])
            print(f"Loaded market data: {len(market_data)} records")
        
        # Load company info
        info_files = [f for f in os.listdir('.') if f.startswith('corp_info_') and f.endswith('.csv')]
        if info_files:
            company_info = pd.read_csv(info_files[0])
            print(f"Loaded company info: {len(company_info)} records")
            
    except Exception as e:
        print(f"Error loading CSV data: {e}")

@app.on_event("startup")
async def startup_event():
    load_csv_data()

@app.get("/")
async def root():
    return {"message": "Financial Data API", "version": "1.0.0"}

@app.get("/api/indicators", response_model=List[EconomicIndicator])
async def search_indicators(q: str = Query(..., description="Search query")):
    """Search economic indicators from IMF and World Bank data"""
    results = []
    
    try:
        # Search IMF data
        if imf_data is not None:
            imf_matches = imf_data[
                imf_data['indicator_name_fr'].str.contains(q, case=False, na=False) |
                imf_data['country'].str.contains(q, case=False, na=False)
            ].head(10)
            
            for _, row in imf_matches.iterrows():
                results.append(EconomicIndicator(
                    id=str(row['id']),
                    title=row['indicator_name_fr'],
                    description=row.get('description', '') or '',
                    source="IMF",
                    category=["macro", "economic"],
                    country=row.get('country', ''),
                    value=str(row.get('value', 'N/A')),
                    lastUpdate=row.get('last_updated', ''),
                    frequency="Annual",
                    year=str(row.get('year', '')),
                    unit=""
                ))
        
        # Search World Bank data
        if wb_data is not None:
            wb_matches = wb_data[
                wb_data['indicator_name_fr'].str.contains(q, case=False, na=False) |
                wb_data['country_name'].str.contains(q, case=False, na=False)
            ].head(10)
            
            for _, row in wb_matches.iterrows():
                results.append(EconomicIndicator(
                    id=str(row['id']),
                    title=row['indicator_name_fr'],
                    description=row.get('indicator_name', '') or '',
                    source="World Bank",
                    category=["economic", "development"],
                    country=row.get('country_name', ''),
                    value=str(row.get('value', 'N/A')),
                    lastUpdate=row.get('last_updated', ''),
                    frequency="Annual",
                    year=str(row.get('year', '')),
                    unit=row.get('unit', '') or ''
                ))
        
    except Exception as e:
        print(f"Error searching indicators: {e}")
        raise HTTPException(status_code=500, detail="Error searching indicators")
    
    return results[:20]  # Limit to 20 results

@app.get("/api/financial/{symbol}", response_model=FinancialData)
async def get_financial_data(symbol: str):
    """Get financial data for a specific symbol"""
    if market_data is None:
        raise HTTPException(status_code=404, detail="Market data not available")
    
    try:
        # Get latest data for symbol
        symbol_data = market_data[market_data['ticker'] == symbol.upper()]
        if symbol_data.empty:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        
        latest = symbol_data.iloc[-1]
        
        return FinancialData(
            symbol=symbol.upper(),
            name=get_company_name(symbol.upper()),
            price=float(latest['close_price']),
            change=float(latest.get('rendement', 0)),
            changePercent=float(latest.get('rendement', 0)),
            volume=float(latest.get('volume', 0)),
            marketCap=float(latest.get('market_cap', 0)) if pd.notna(latest.get('market_cap')) else None
        )
    
    except Exception as e:
        print(f"Error getting financial data: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving financial data")

@app.get("/api/chart/{symbol}", response_model=List[ChartData])
async def get_chart_data(symbol: str, period: str = Query("1M", description="Time period")):
    """Get chart data for a specific symbol"""
    if market_data is None:
        raise HTTPException(status_code=404, detail="Market data not available")
    
    try:
        symbol_data = market_data[market_data['ticker'] == symbol.upper()]
        if symbol_data.empty:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")
        
        # Sort by timestamp and limit based on period
        symbol_data = symbol_data.sort_values('timestamp')
        
        # Simple period filtering (you can enhance this)
        if period == "1W":
            symbol_data = symbol_data.tail(7)
        elif period == "1M":
            symbol_data = symbol_data.tail(30)
        elif period == "3M":
            symbol_data = symbol_data.tail(90)
        elif period == "1Y":
            symbol_data = symbol_data.tail(365)
        
        chart_data = []
        for _, row in symbol_data.iterrows():
            chart_data.append(ChartData(
                timestamp=row['timestamp'],
                open=float(row['open_price']),
                high=float(row['high_price']),
                low=float(row['low_price']),
                close=float(row['close_price']),
                volume=float(row['volume'])
            ))
        
        return chart_data
    
    except Exception as e:
        print(f"Error getting chart data: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving chart data")

@app.get("/api/companies", response_model=List[CompanyInfo])
async def get_companies(limit: int = Query(10, description="Number of companies to return")):
    """Get list of companies"""
    if company_info is None:
        raise HTTPException(status_code=404, detail="Company data not available")
    
    try:
        companies = []
        for _, row in company_info.head(limit).iterrows():
            companies.append(CompanyInfo(
                ticker=row['ticker'],
                company_name=row.get('company_name', ''),
                long_business_summary=row.get('long_business_summary', ''),
                website=row.get('website', ''),
                industry=row.get('industry', ''),
                sector=row.get('sector', ''),
                country=row.get('country', ''),
                city=row.get('city', ''),
                employees=int(row['full_time_employees']) if pd.notna(row.get('full_time_employees')) else None
            ))
        
        return companies
    
    except Exception as e:
        print(f"Error getting companies: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving companies")

@app.get("/api/company/{ticker}", response_model=CompanyInfo)
async def get_company(ticker: str):
    """Get specific company information"""
    if company_info is None:
        raise HTTPException(status_code=404, detail="Company data not available")
    
    try:
        company_data = company_info[company_info['ticker'] == ticker.upper()]
        if company_data.empty:
            raise HTTPException(status_code=404, detail=f"Company {ticker} not found")
        
        row = company_data.iloc[0]
        return CompanyInfo(
            ticker=row['ticker'],
            company_name=row.get('company_name', ''),
            long_business_summary=row.get('long_business_summary', ''),
            website=row.get('website', ''),
            industry=row.get('industry', ''),
            sector=row.get('sector', ''),
            country=row.get('country', ''),
            city=row.get('city', ''),
            employees=int(row['full_time_employees']) if pd.notna(row.get('full_time_employees')) else None
        )
    
    except Exception as e:
        print(f"Error getting company: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving company")

@app.get("/api/economic/{indicator_id}")
async def get_economic_indicator(indicator_id: str):
    """Get specific economic indicator data"""
    try:
        # Search in IMF data
        if imf_data is not None:
            imf_result = imf_data[imf_data['id'] == int(indicator_id)]
            if not imf_result.empty:
                row = imf_result.iloc[0]
                return {
                    "id": str(row['id']),
                    "title": row['indicator_name_fr'],
                    "description": row.get('description', ''),
                    "source": "IMF",
                    "value": row.get('value', 'N/A'),
                    "country": row.get('country', ''),
                    "year": row.get('year', ''),
                    "lastUpdate": row.get('last_updated', '')
                }
        
        # Search in World Bank data
        if wb_data is not None:
            wb_result = wb_data[wb_data['id'] == int(indicator_id)]
            if not wb_result.empty:
                row = wb_result.iloc[0]
                return {
                    "id": str(row['id']),
                    "title": row['indicator_name_fr'],
                    "description": row.get('indicator_name', ''),
                    "source": "World Bank",
                    "value": row.get('value', 'N/A'),
                    "country": row.get('country_name', ''),
                    "year": row.get('year', ''),
                    "lastUpdate": row.get('last_updated', '')
                }
        
        raise HTTPException(status_code=404, detail="Indicator not found")
        
    except Exception as e:
        print(f"Error getting economic indicator: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving indicator")

def get_company_name(ticker: str) -> Optional[str]:
    """Helper function to get company name from ticker"""
    if company_info is not None:
        company_data = company_info[company_info['ticker'] == ticker]
        if not company_data.empty:
            return company_data.iloc[0].get('company_name', ticker)
    return ticker

@app.get("/api/search")
async def search_all(q: str = Query(..., description="Search query")):
    """Global search across all data sources"""
    results = {
        "indicators": [],
        "companies": [],
        "symbols": []
    }
    
    try:
        # Search indicators
        indicators_response = await search_indicators(q)
        results["indicators"] = [indicator.dict() for indicator in indicators_response[:5]]
        
        # Search companies
        if company_info is not None:
            company_matches = company_info[
                company_info['company_name'].str.contains(q, case=False, na=False) |
                company_info['ticker'].str.contains(q, case=False, na=False)
            ].head(5)
            
            for _, row in company_matches.iterrows():
                results["companies"].append({
                    "ticker": row['ticker'],
                    "name": row.get('company_name', ''),
                    "sector": row.get('sector', ''),
                    "industry": row.get('industry', '')
                })
        
        # Search market symbols
        if market_data is not None:
            symbol_matches = market_data[
                market_data['ticker'].str.contains(q, case=False, na=False)
            ]['ticker'].unique()[:5]
            
            for symbol in symbol_matches:
                results["symbols"].append({
                    "symbol": symbol,
                    "name": get_company_name(symbol)
                })
        
    except Exception as e:
        print(f"Error in global search: {e}")
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
