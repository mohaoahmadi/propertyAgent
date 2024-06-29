import streamlit as st
import numpy as np
import pandas as pd
from daftlistings import Daft, SearchType, PropertyType, SortType, Distance, Location, MapVisualization
import logging
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def calculate_mortgage(price, down_payment, interest_rate, loan_term):
    logger.debug(f"Calculating mortgage: price={price}, down_payment={down_payment}, interest_rate={interest_rate}, loan_term={loan_term}")
    try:
        loan_amount = price - down_payment
        monthly_interest_rate = interest_rate / 12 / 100
        num_payments = loan_term * 12
        monthly_payment = (loan_amount * monthly_interest_rate * (1 + monthly_interest_rate)**num_payments) / ((1 + monthly_interest_rate)**num_payments - 1)
        return monthly_payment
    except Exception as e:
        logger.error(f"Error in mortgage calculation: {str(e)}")
        return 0

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
        'daft_link': mapping_dict.get('daft_link', 'N/A'),
        'latitude': mapping_dict.get('latitude'),
        'longitude': mapping_dict.get('longitude'),
    }

def main():
    st.set_page_config(layout="wide")
    st.title("Property Search App")

    # Sidebar for inputs
    st.sidebar.header("Mortgage Calculator")
    max_price = st.sidebar.number_input("Maximum Property Price (€)", min_value=0, value=500000, step=10000)
    down_payment = st.sidebar.number_input("Down Payment (€)", min_value=0, value=50000, step=1000)
    interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
    loan_term = st.sidebar.number_input("Loan Term (years)", min_value=5, max_value=35, value=30, step=5)

    monthly_payment = calculate_mortgage(max_price, down_payment, interest_rate, loan_term)
    st.sidebar.write(f"Estimated Monthly Payment: €{monthly_payment:.2f}")

    st.sidebar.header("Search Criteria")
    min_beds = st.sidebar.number_input("Minimum Bedrooms", min_value=1, value=2)
    min_baths = st.sidebar.number_input("Minimum Bathrooms", min_value=1, value=1)
    min_price = st.sidebar.number_input("Minimum Price (€)", min_value=0, value=0, step=10000)
    min_floor_size = st.sidebar.number_input("Minimum Floor Size (sqm)", min_value=0, value=0, step=10)
    location = st.sidebar.selectbox("Location", [loc.name for loc in Location])
    property_type = st.sidebar.multiselect("Property Type", [pt.name for pt in PropertyType])
    search_type = st.sidebar.radio("Search Type", ["Residential Sale", "New Homes"])

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
                st.write(display_df[['title', 'price', 'bedrooms', 'bathrooms', 'floor_size', 'daft_link']].to_html(justify='center', escape=False, index=True), unsafe_allow_html=True)
                
                # Clean data for map visualization
                map_data = df[['latitude', 'longitude', 'price', 'bedrooms', 'bathrooms', 'daft_link']].copy()
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