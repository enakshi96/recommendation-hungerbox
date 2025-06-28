import pandas as pd
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")
# Load CSV with warning handled
orders = pd.read_csv("accenture_sales_order.csv", low_memory=False)
order_items = pd.read_csv("accenture_order_items.csv")

# Check shapes
print(f"Orders shape: {orders.shape}")
print(f"Order Items shape: {order_items.shape}")

# Check column names (debug)
print("Orders columns:", orders.columns.tolist())

# Rename 'id' to 'order_id_master' to avoid conflict during merge
orders_renamed = orders.rename(columns={"id": "order_id_master"})

# Now make sure 'status' is there
print("Renamed Orders columns:", orders_renamed.columns.tolist())

# Merge: join order_items with required columns from orders
merged_df = order_items.merge(
    orders_renamed[['order_id_master', 'employee_id', 'status']],
    left_on='order_id',
    right_on='order_id_master',
    how='inner'
)

# Check merged columns
print("Merged columns:", merged_df.columns.tolist())

# Filter only delivered orders
delivered_df = merged_df[merged_df['status_y'] == 'delivered'].copy()
delivered_df.drop(columns=['order_id_master', 'status_x', 'status_y'], inplace=True)

# Drop extra id column
delivered_df.drop(columns=['order_id_master'], errors='ignore', inplace=True)
print("Delivered columns:", delivered_df.columns.tolist())
delivered_df = delivered_df[['employee_id', 'product_id', 'order_id', 'product_name', 'qty']]

# Show sample
print("Cleaned Delivered Data:")
print(delivered_df.head())
# Preview result
interaction_df = (
    delivered_df
    .groupby(['employee_id', 'product_id', 'product_name'])
    .size()
    .reset_index(name='order_count')
)

# Step 2: Cap count at 5
interaction_df['order_count'] = interaction_df['order_count'].clip(upper=5)
# Remove rows where product_name is exactly "Pav One Piece"
blacklist=["Pav 1pcs", "Pav 1Pcs", "Roti", "Chapathi"]


interaction_df = interaction_df[~interaction_df['product_name'].isin(blacklist)]

# Show result
print("Sample interactions (capped at 5):")
print(interaction_df.head())




# Step 2: Prepare data for surprise
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(interaction_df[['employee_id', 'product_id', 'order_count']], reader)

trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Step 3: Train SVD model
model = SVD()
model.fit(trainset)
# Create a lookup dictionary from product_id to product_name
product_lookup = interaction_df.drop_duplicates('product_id').set_index('product_id')['product_name'].to_dict()


@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, employee_id: int = Form(...)):
    ordered = interaction_df[interaction_df['employee_id'] == employee_id]
    ordered_product_ids = set(ordered['product_id'])
    repeat_items = ordered.sort_values('order_count', ascending=False).to_dict('records')

    all_products = interaction_df['product_id'].unique()
    recommendations = []
    for pid in all_products:
        if pid not in ordered_product_ids:
            pred = model.predict(employee_id, pid)
            recommendations.append((pid, pred.est))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    reco_items = [{"name": product_lookup.get(pid, "Unknown"), "score": round(score, 2)}
                  for pid, score in recommendations[:5]]

    return templates.TemplateResponse("result.html", {
        "request": request,
        "employee_id": employee_id,
        "repeat_items": repeat_items,
        "recommendations": reco_items
    })