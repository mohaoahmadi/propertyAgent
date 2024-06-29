import streamlit as st
import pandas as pd
import numpy as np
from daftlistings import Daft, SearchType, PropertyType, SortType, Distance, Location, MapVisualization, Ber
import logging
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def calculate_mortgage(price, down_payment, interest_rate, loan_term, property_tax_rate=0, insurance_cost=0, is_green_mortgage=False):
    logger.debug(f"Calculating mortgage: price={price}, down_payment={down_payment}, interest_rate={interest_rate}, loan_term={loan_term}")
    try:
        loan_amount = price - down_payment
        monthly_interest_rate = interest_rate / 12 / 100
        num_payments = loan_term * 12
        
        # Apply green mortgage discount if applicable
        if is_green_mortgage:
            monthly_interest_rate *= 0.95  # 5% discount for green mortgages
        
        monthly_payment = (loan_amount * monthly_interest_rate * (1 + monthly_interest_rate)**num_payments) / ((1 + monthly_interest_rate)**num_payments - 1)
        
        # Add property tax and insurance
        monthly_payment += (price * property_tax_rate / 12) + (insurance_cost / 12)
        
        total_cost = monthly_payment * num_payments
        total_interest = total_cost - loan_amount
        
        return {
            'monthly_payment': monthly_payment,
            'total_cost': total_cost,
            'total_interest': total_interest,
            'loan_to_value_ratio': (loan_amount / price) * 100
        }
    except Exception as e:
        logger.error(f"Error in mortgage calculation: {str(e)}")
        return None

def get_mortgage_options(price, down_payment, loan_term):
    options = [
        {'name': 'AIB', 'rate': 3.45, 'term': 4, 'is_green': True},
        {'name': 'Bank of Ireland', 'rate': 3.6, 'term': 4, 'is_green': True},
        {'name': 'Avant Money', 'rate': 3.8, 'term': 4, 'is_green': False},
        {'name': 'Avant Money', 'rate': 3.95, 'term': 7, 'is_green': False},
    ]
    
    results = []
    for option in options:
        result = calculate_mortgage(price, down_payment, option['rate'], loan_term, is_green_mortgage=option['is_green'])
        if result:
            results.append({
                'provider': option['name'],
                'rate': option['rate'],
                'term': option['term'],
                'is_green': option['is_green'],
                'monthly_payment': result['monthly_payment'],
                'total_cost': result['total_cost'],
                'total_interest': result['total_interest'],
                'loan_to_value_ratio': result['loan_to_value_ratio']})
    
    return results

def safe_get(obj, key, default=None):
    try:
        return obj[key]
    except (KeyError, TypeError):
        return default

def parse_price(price_str):
    if isinstance(price_str, (int, float)):
        return float(price_str)
    if isinstance(price_str, str):
        # Remove currency symbols and commas
        price_str = re.sub(r'[€,]', '', price_str)
        # Extract the numeric part
        match = re.search(r'\d+', price_str)
        if match:
            return float(match.group())
    return np.nan

def process_listing(listing):
    logger.debug(f"Processing listing: {listing}")
    full_dict = listing.as_dict()
    mapping_dict = listing.as_dict_for_mapping()
    return {
        'title': safe_get(full_dict, 'title', 'N/A'),
        'price': parse_price(safe_get(full_dict, 'price', 'N/A')),
        'monthly_price': parse_price(mapping_dict.get('monthly_price', 'N/A')),
        'bedrooms': mapping_dict.get('bedrooms', 'N/A'),
        'bathrooms': mapping_dict.get('bathrooms', 'N/A'),
        'floor_size': safe_get(full_dict.get('floorArea', {}), 'value', 'N/A'),
        'ber_rating': safe_get(full_dict.get('ber', {}), 'rating', 'N/A'),
        'daft_link': mapping_dict.get('daft_link', 'N/A'),
        'latitude': mapping_dict.get('latitude'),
        'longitude': mapping_dict.get('longitude'),
    }

def main():
    st.set_page_config(layout="wide")
    st.title("Property Search and Mortgage Calculator App")

    # Sidebar for inputs
    st.sidebar.header("Mortgage Calculator")
    price = st.sidebar.number_input("Property Price (€)", min_value=0, value=350000, step=10000)
    down_payment = st.sidebar.number_input("Down Payment (€)", min_value=0, value=50000, step=1000)
    loan_term = st.sidebar.number_input("Loan Term (years)", min_value=5, max_value=35, value=30, step=5)
    property_tax_rate = st.sidebar.number_input("Annual Property Tax Rate (%)", min_value=0.0, max_value=5.0, value=0.18, step=0.01)
    insurance_cost = st.sidebar.number_input("Annual Insurance Cost (€)", min_value=0, value=1000, step=100)
    is_green_mortgage = st.sidebar.checkbox("Apply for Green Mortgage")

    # Calculate mortgage options
    mortgage_options = get_mortgage_options(price, down_payment, loan_term)

    # Display mortgage comparison
    st.header("Mortgage Comparison")
    mortgage_df = pd.DataFrame(mortgage_options)
    mortgage_df['monthly_payment'] = mortgage_df['monthly_payment'].apply(lambda x: f"€{x:.2f}")
    mortgage_df['total_cost'] = mortgage_df['total_cost'].apply(lambda x: f"€{x:.2f}")
    mortgage_df['total_interest'] = mortgage_df['total_interest'].apply(lambda x: f"€{x:.2f}")
    mortgage_df['loan_to_value_ratio'] = mortgage_df['loan_to_value_ratio'].apply(lambda x: f"{x:.2f}%")
    mortgage_df['is_green'] = mortgage_df['is_green'].apply(lambda x: 'Yes' if x else 'No')
    st.table(mortgage_df)

    # Property Search
    st.sidebar.header("Property Search Criteria")
    min_beds = st.sidebar.number_input("Minimum Bedrooms", min_value=1, value=2)
    min_baths = st.sidebar.number_input("Minimum Bathrooms", min_value=1, value=1)
    min_price = st.sidebar.number_input("Minimum Price (€)", min_value=0, value=0, step=10000)
    max_price = st.sidebar.number_input("Maximum Price (€)", min_value=0, value=500000, step=10000)
    min_floor_size = st.sidebar.number_input("Minimum Floor Size (sqm)", min_value=0, value=0, step=10)
    location = st.sidebar.selectbox("Location", [loc.name for loc in Location])
    property_type = st.sidebar.multiselect("Property Type", [pt.name for pt in PropertyType])
    search_type = st.sidebar.radio("Search Type", ["Residential Sale", "New Homes"])
    
    # Add BER Energy Rating to search criteria
    ber_options = ['Any'] + [ber.name for ber in Ber]
    min_ber_rating = st.sidebar.selectbox("Minimum BER Energy Rating", ber_options)
    max_ber_rating = st.sidebar.selectbox("Maximum BER Energy Rating", ber_options)

    if st.sidebar.button("Search Properties"):
        try:
            daft = Daft()
            daft.set_search_type(SearchType.RESIDENTIAL_SALE if search_type == "Residential Sale" else SearchType.NEW_HOMES)
            daft.set_location(getattr(Location, location))
            daft.set_min_price(min_price)
            daft.set_max_price(max_price)
            daft.set_min_beds(min_beds)
            daft.set_min_baths(min_baths)
            daft.set_min_floor_size(min_floor_size)
            for pt in property_type:
                daft.set_property_type(getattr(PropertyType, pt))
            daft.set_sort_type(SortType.PRICE_ASC)
            
            # Set BER range
            if min_ber_rating != 'Any':
                daft.set_min_ber(getattr(Ber, min_ber_rating))
            if max_ber_rating != 'Any':
                daft.set_max_ber(getattr(Ber, max_ber_rating))

            listings = daft.search()
            
            if listings:
                processed_listings = [process_listing(listing) for listing in listings]
                df = pd.DataFrame(processed_listings)
                
                # Display results in a table
                st.subheader("Search Results")
                display_df = df.copy()
                display_df['price'] = display_df['price'].apply(lambda x: f"€{x:,.0f}" if pd.notnull(x) else 'N/A')
                display_df['floor_size'] = display_df['floor_size'].apply(lambda x: f"{x} sqm" if x != 'N/A' else 'N/A')
                display_df['daft_link'] = display_df['daft_link'].apply(lambda x: f'<a href="{x}" target="_blank">View</a>' if x != 'N/A' else 'N/A')
                st.write(display_df[['title', 'price', 'bedrooms', 'bathrooms', 'floor_size', 'ber_rating', 'daft_link']].to_html(escape=False, index=False), unsafe_allow_html=True)
                
                # Clean data for map visualization
                map_data = df[['latitude', 'longitude', 'price', 'bedrooms', 'bathrooms', 'ber_rating', 'daft_link']].copy()
                map_data['price'] = pd.to_numeric(map_data['price'], errors='coerce')
                map_data = map_data.dropna(subset=['latitude', 'longitude', 'price'])
                
                if not map_data.empty:
                    # Ensure 'daft_link' is present, use a placeholder if missing
                    if 'daft_link' not in map_data.columns:
                        map_data['daft_link'] = 'N/A'
                    
                    map_vis = MapVisualization(map_data)
                    map_vis.add_markers()
                    map_vis.add_colorbar()
                    st.components.v1.html(map_vis.map._repr_html_(), height=600)
                else:
                    st.write("Not enough location data to display map.")
            else:
                st.write("No properties found matching your criteria.")
        except Exception as e:
            logger.exception("An error occurred while searching for properties")
            st.error(f"An error occurred while searching for properties: {str(e)}")

if __name__ == "__main__":
    main()