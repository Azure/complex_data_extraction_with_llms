
# Getting Started

### 1. Setting Up Azure Resources
#### 1.1 Prerequisites
    1. Azure account: Create azure account by [signing up here](https://azure.microsoft.com/)
    2. Azure CLI: Install the Azure CLI from [here](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli).

#### 1.2 Steps
##### 1.2.1 Login to Azure
Open your terminal and login to your Azure account:
```bash
az login
```
Follow the instructions to complete the authentication process. If you are using a specific subscription, set it as the default:
```bash
az account set --subscription "your-subscription-id"
```

#### 1.2.2 Creating needed resources
##### 1.2.2.1 Create a resource group (if you don't have one):
``` bash
az group create --name <resource_group_name> --location <region>
```
##### 1.2.2.2 Create azure openAI resource
1. In your Azure portal, click on “Create a resource”.
2. Search for “OpenAI” and select it.
3. Click on “Create” and fill in the necessary details such as name, subscription, resource group, etc.
4. Click on “Review + Create” and then “Create” to create the resource.
5. Once the deployment is complete, go to the resource page.
6. Under “Keys and Endpoint”, you can find your key and endpoint. Save these for later use.

##### 1.2.2.3 Create Document Intelligence Resource
1. In your Azure portal, click on “Create a resource”.
2. Search for “Document Intelligence” and select it.
3. Click on “Create” and fill in the necessary details.
4. Click on “Review + Create” and then “Create” to create the resource.
5. Once the deployment is complete, go to the resource page.
6. Under “Keys and Endpoint”, you can find your key and endpoint. Save these for later use.

##### 1.2.2.4 Create Azure Search Resource
1. In your Azure portal, click on “Create a resource”.
2. Search for “Azure Search” and select it.
3. Click on “Create” and fill in the necessary details.
4. Click on “Review + Create” and then “Create” to create the resource.
5. Once the deployment is complete, go to the resource page.
6. Under “Keys and Endpoint”, you can find your key and endpoint. Save these for later use.

##### 1.2.2.5 Get the Azure Tenant ID
1. In your Azure portal, Click on Azure Active Directory in the left-hand menu.
3. Your Tenant ID is listed as Directory ID on the default page.

##### 1.2.2.6 Get the Azure Client ID and Client Secret
1. In the Azure portal, click on App Registrations in the left-hand menu under Azure Active Directory.
3. Click on New Registration at the top.
5. Fill in the details such as name, supported account types, and redirect URI (if necessary), then click Register.
7. After the app is registered, the Application (client) ID is displayed on the app page. This is your Client ID.
9. To get the Client Secret, click on Certificates & secrets in the left-hand menu of the app page.
11. Click on New client secret, add a description, select an expiry period, and click Add.
13. After the client secret is created, copy the Value. This is your Client Secret.

### 2. Create environment variables in .env file 
# 2.1 Creating a .env file
1. Open your code editor or terminal.
2. Navigate to the root directory of your project.
3. Create a new file named `.env`.
4. Open the `.env` file.
5. Add your environment variables in the format `KEY=VALUE`, one per line. For example:

   ```bash
   AZURE_SUBSCRIPTION_ID=<your-subscription-id>
   AZURE_TENANT_ID=<your-tenant-id>
   AZURE_CLIENT_ID=<your-client-id>
   AZURE_CLIENT_SECRET=<your-client-secret>
   AZURE_OPENAI_API_KEY=<your-openai-key>
   AZURE_OPENAI_ENDPOINT=<your-openai-endpoint>
   DOC_INTELLIGENCE_KEY=<your-document-intelligence-key>
   DOC_INTELLIGENCE_ENDPOINT=<your-document-intelligence-endpoint>
   VECTOR_SEARCH_KEY=<your-azure-search-key>
   VECTOR_SEARCH_ENDPOINT=<your-azure-search-endpoint>
   ```

### 3. Creating a virtual environment
It is recommended that Python virtual environments are used for local branch development.
Then main advantage of using virtual environments is that you can create a separate workspace environment for a branch, so that yo can safely install, remove or upgrade a library without affecting other environments.
`venv` docs: https://docs.python.org/3/library/venv.html
Create a new environment with Python version=3.11
``` bash
conda create -n avalara python==3.11
```
Then, activate the environment
```bash
conda activate avalara
```

### 4. Installing dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the notebook
Refer to the notebook for initiating the Q&A process with your documents baseline.
You may need to create a folder named data if not already created.