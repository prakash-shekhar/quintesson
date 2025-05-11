// App.js
import React, { useState } from 'react';
import { Info, Plus, Edit } from 'lucide-react';
import './App.css';

const PeftApp = () => {
  const [models, setModels] = useState([
    {
      id: 1,
      name: 'Custom LLama',
      baseModel: 'Llama 3.1',
      status: 'ready',
      settings: {
        quantization: '8bit',
        device: 'cuda',
        cloudProvider: 'local',
        modelId: 'meta-llama/Meta-Llama-3.1-8B'
      }
    }
  ]);

  const [selectedModel, setSelectedModel] = useState(models[0]);
  const [activeTab, setActiveTab] = useState('peft');
  const [showNewLLMModal, setShowNewLLMModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [modelToEdit, setModelToEdit] = useState(null);
  
  // Add a function to handle adding new models
  const handleAddModel = (newModel) => {
    setModels([...models, newModel]);
    setSelectedModel(newModel);
    setShowNewLLMModal(false);
  };
  
  // Add function to handle editing models
  const handleEditModel = (editedModel) => {
    setModels(models.map(model => 
      model.id === editedModel.id ? editedModel : model
    ));
    setSelectedModel(editedModel);
    setShowEditModal(false);
    setModelToEdit(null);
  };
  
  // Function to open edit modal
  const openEditModal = (model) => {
    setModelToEdit(model);
    setShowEditModal(true);
  };

  const modelTabs = [
    {
      id: 'peft',
      title: 'PEFT',
      description: 'Parameter-Efficient Fine Tuning (most popular approach for fine-tuning aLLMs)'
    },
    {
      id: 'rome',
      title: 'MEMIT',
      description: 'Mass-Edit Memory in Transformer (Inject Facts into the LLM)'
    },
    {
      id: 'coar',
      title: 'CoAR',
      description: 'Ablates components of the LLM'
    },
    {
      id: 'harmbench',
      title: 'HarmBench',
      description: 'Test your LLM against cutting-edge red-teaming'
    }
  ];

  const ModelCard = ({ model, isSelected, onClick, onEdit }) => (
    <div 
      className={`model-card ${isSelected ? 'selected' : ''}`}
      onClick={() => onClick(model)}
    >
      <div className="model-card-content">
        <div className="model-icon">
          <div className="llama-icon">
            <div className="icon-face"></div>
            <div className="icon-eye left"></div>
            <div className="icon-eye right"></div>
            <div className="icon-mouth"></div>
          </div>
        </div>
        <div className="model-info">
          <h3>{model.name}</h3>
          <p>{model.baseModel}</p>
          {model.settings && (
            <div className="model-settings">
              <span className="setting-badge">{model.settings.quantization}</span>
              <span className="setting-badge">{model.settings.device}</span>
              <span className="setting-badge">{model.settings.cloudProvider}</span>
            </div>
          )}
          <button 
            className="edit-btn"
            onClick={(e) => {
              e.stopPropagation();
              onEdit(model);
            }}
          >
            Edit
          </button>
        </div>
      </div>
    </div>
  );

  const TabContent = ({ tab }) => {
    if (tab.id === 'peft') {
      return (
        <div className="tab-content">
          <div className="config-section">
            <h3>PEFT Configuration</h3>
            <div className="form-group">
              <label>PEFT Method</label>
              <select>
                <option>LoRA</option>
                <option>QLoRA</option>
                <option>Prefix Tuning</option>
                <option>Adapter Layers</option>
              </select>
            </div>
  
            <div className="form-row">
              <div className="form-group">
                <label>
                  Rank <Info size={14} className="inline-icon" />
                </label>
                <input type="number" defaultValue={8} min={1} max={64} />
                <p className="help-text">How much the model is allowed to learn. Higher = more capacity to learn new info. Lower = small tweaks (e.g. style).</p>
              </div>
              <div className="form-group">
                <label>
                  Alpha <Info size={14} className="inline-icon" />
                </label>
                <input type="number" defaultValue={16} min={1} max={64} />
                <p className="help-text">How strong the new learning is applied. Usually set equal to Rank. Higher = more forceful changes.</p>
              </div>
            </div>
  
            <div className="form-group">
              <label>
                Target Modules (comma-separated) <Info size={14} className="inline-icon" />
              </label>
              <input type="text" placeholder="q_proj,v_proj,k_proj,o_proj" />
              <p className="help-text">Which parts of the model to modify. Defaults: <code>q_proj,v_proj,k_proj,o_proj</code> — common attention layers.</p>
            </div>
          </div>
  
          <div className="config-section">
            <h3>Training Configuration</h3>
            <div className="form-row">
              <div className="form-group">
                <label>
                  Learning Rate <Info size={14} className="inline-icon" />
                </label>
                <input type="number" defaultValue={0.0002} step={0.00001} />
                <p className="help-text">How fast the model learns. Too high = messy learning. Too low = very slow training.</p>
              </div>
              <div className="form-group">
                <label>
                  Epochs <Info size={14} className="inline-icon" />
                </label>
                <input type="number" defaultValue={3} min={1} max={10} />
                <p className="help-text">How many times your data is used. More = better learning, but longer training.</p>
              </div>
            </div>
  
            <div className="form-row">
              <div className="form-group">
                <label>
                  Batch Size <Info size={14} className="inline-icon" />
                </label>
                <input type="number" defaultValue={4} min={1} max={32} />
                <p className="help-text">How much data the model sees at once. Smaller = less memory use, slower training. Bigger = faster, but uses more VRAM.</p>
              </div>
              <div className="form-group">
                <label>
                  Max Sequence Length <Info size={14} className="inline-icon" />
                </label>
                <input type="number" defaultValue={512} min={64} max={2048} />
                <p className="help-text">Longest text input your model can see. Too high uses more memory. Too low may cut off your examples.</p>
              </div>
            </div>
          </div>
  
          <div className="config-section">
            <h3>Training Data</h3>
            <div className="drop-zone">
              <div className="drop-zone-content">
                <p>Drop your dataset here or click to upload</p>
                <p className="drop-zone-hint">Supports JSON, JSONL, CSV formats</p>
              </div>
              <input type="file" className="file-input" accept=".json,.jsonl,.csv" />
            </div>
  
            <div className="form-group">
              <label>Or paste your data directly</label>
              <textarea
                className="data-textarea"
                placeholder='{"instruction": "Explain PEFT", "response": "PEFT is..."}'
              />
            </div>
          </div>
  
          <div className="actions">
            <button className="primary-btn">Start Training</button>
          </div>
        </div>
      );
    } else if (tab.id === 'rome') {
      return (
        <div className="tab-content">
          <div className="config-section">
            <h3>MEMIT Configuration</h3>
            <div className="form-group">
              <label>
                Algorithm <Info size={14} className="inline-icon" />
              </label>
              <select>
                <option>MEMIT</option>
                <option>ROME</option>
                <option>FT</option>
                <option>FT-L</option>
                <option>FT-AttnEdit</option>
                <option>MEND</option>
                <option>MEND-CF</option>
                <option>MEND-zsRE</option>
              </select>
              <p className="help-text">MEMIT: Mass-Editing Memory in a Transformer (recommended for multiple edits). ROME: Rank-One Model Editing. FT: Fine-Tuning variants. MEND: Hypernetwork-based editing.</p>
            </div>
  
            <div className="form-group">
              <label>
                Number of Edits <Info size={14} className="inline-icon" />
              </label>
              <input type="number" defaultValue={1} min={1} max={100} />
              <p className="help-text">How many facts to edit at once. MEMIT excels at batch editing multiple facts simultaneously.</p>
            </div>
          </div>
  
          <div className="config-section">
            <h3>Fact Editing</h3>
            
            <div className="form-group">
              <label>Edit Mode</label>
              <select>
                <option>Single Fact Editor</option>
                <option>Bulk JSON Upload</option>
              </select>
            </div>
            
            <div className="fact-editor">
              <div className="fact-item">
                <div className="form-group">
                  <label>
                    Subject <Info size={14} className="inline-icon" />
                  </label>
                  <input type="text" placeholder="Steve Jobs" />
                  <p className="help-text">The entity you want to change information about.</p>
                </div>
  
                <div className="form-group">
                  <label>
                    Fact Template <Info size={14} className="inline-icon" />
                  </label>
                  <input type="text" placeholder="{} was the founder of" />
                  <p className="help-text">Template with <code>{}</code> placeholder for the subject. Example: <code>{} was the founder of</code></p>
                </div>
  
                <div className="form-group">
                  <label>
                    New Target <Info size={14} className="inline-icon" />
                  </label>
                  <input type="text" placeholder="Microsoft" />
                  <p className="help-text">What you want the new fact to be. Example: If the template is <code>{} was the founder of</code> and target is <code>Microsoft</code>, the model will say "Steve Jobs was the founder of Microsoft".</p>
                </div>
              </div>
  
              <button className="add-fact-btn">
                <Plus size={16} />
                Add Another Fact
              </button>
            </div>
            
            <div className="config-section">
              <h4>Or upload multiple facts:</h4>
              <div className="drop-zone">
                <div className="drop-zone-content">
                  <p>Drop your facts file here or click to upload</p>
                  <p className="drop-zone-hint">Supports JSON format</p>
                </div>
                <input type="file" className="file-input" accept=".json" />
              </div>
              
              <div className="form-group">
                <label>Or paste your facts directly</label>
                <textarea
                  className="data-textarea"
                  placeholder='[
  {
    "subject": "Steve Jobs",
    "prompt": "{} was the founder of",
    "target_new": "Microsoft"
  },
  {
    "subject": "LeBron James",
    "prompt": "{} plays the sport of",
    "target_new": "football"
  }
]'
                />
              </div>
            </div>
          </div>
  
          <div className="actions">
            <button className="primary-btn">Edit Facts</button>
          </div>
        </div>
      );
    }
  
    return (
      <div className="empty-tab">
        <h3>{tab.title}</h3>
        <p>{tab.description}</p>
        <div className="coming-soon">
          <p>Coming soon...</p>
        </div>
      </div>
    );
  };
  
  const NewLLMModal = ({ isOpen, onClose, onAddModel, editModel = null }) => {
    const [selectedModel, setSelectedModel] = useState('');
    const [selectedVersion, setSelectedVersion] = useState('');
    const [modelName, setModelName] = useState('');
    const [quantization, setQuantization] = useState('8bit');
    const [device, setDevice] = useState('cuda');
    const [cloudProvider, setCloudProvider] = useState('local');
    
    // Pre-populate form when editing
    React.useEffect(() => {
      if (editModel) {
        setModelName(editModel.name);
        setQuantization(editModel.settings?.quantization || '8bit');
        setDevice(editModel.settings?.device || 'cuda');
        setCloudProvider(editModel.settings?.cloudProvider || 'local');
        
        // Try to find the model family and version
        if (editModel.settings?.modelId) {
          for (const [family, info] of Object.entries(modelOptions)) {
            const version = info.versions.find(v => v.id === editModel.settings.modelId);
            if (version) {
              setSelectedModel(family);
              setSelectedVersion(editModel.settings.modelId);
              break;
            }
          }
        }
      } else {
        // Reset form for new model
        setSelectedModel('');
        setSelectedVersion('');
        setModelName('');
        setQuantization('8bit');
        setDevice('cuda');
        setCloudProvider('local');
      }
    }, [editModel]);

    const modelOptions = {
      'llama': {
        name: 'LLama',
        versions: [
          { id: 'meta-llama/Llama-2-7b-hf', name: 'LLama-2 7B' },
          { id: 'meta-llama/Llama-2-13b-hf', name: 'LLama-2 13B' },
          { id: 'meta-llama/Llama-2-70b-hf', name: 'LLama-2 70B' },
          { id: 'meta-llama/Meta-Llama-3-8B', name: 'LLama-3 8B' },
          { id: 'meta-llama/Meta-Llama-3-70B', name: 'LLama-3 70B' },
          { id: 'meta-llama/Meta-Llama-3.1-8B', name: 'LLama-3.1 8B' },
          { id: 'meta-llama/Meta-Llama-3.1-70B', name: 'LLama-3.1 70B' },
        ]
      },
      'mistral': {
        name: 'Mistral',
        versions: [
          { id: 'mistralai/Mistral-7B-v0.1', name: 'Mistral 7B v0.1' },
          { id: 'mistralai/Mistral-7B-Instruct-v0.1', name: 'Mistral 7B Instruct v0.1' },
          { id: 'mistralai/Mistral-7B-Instruct-v0.2', name: 'Mistral 7B Instruct v0.2' },
          { id: 'mistralai/Mixtral-8x7B-v0.1', name: 'Mixtral 8x7B v0.1' },
          { id: 'mistralai/Mixtral-8x7B-Instruct-v0.1', name: 'Mixtral 8x7B Instruct v0.1' },
        ]
      },
      'phi': {
        name: 'Phi',
        versions: [
          { id: 'microsoft/phi-1_5', name: 'Phi-1.5' },
          { id: 'microsoft/phi-2', name: 'Phi-2' },
          { id: 'microsoft/Phi-3-mini-4k-instruct', name: 'Phi-3 Mini 4K' },
          { id: 'microsoft/Phi-3-medium-4k-instruct', name: 'Phi-3 Medium 4K' },
        ]
      },
      'qwen': {
        name: 'Qwen',
        versions: [
          { id: 'Qwen/Qwen-7B', name: 'Qwen 7B' },
          { id: 'Qwen/Qwen-14B', name: 'Qwen 14B' },
          { id: 'Qwen/Qwen-72B', name: 'Qwen 72B' },
          { id: 'Qwen/Qwen1.5-0.5B', name: 'Qwen1.5 0.5B' },
          { id: 'Qwen/Qwen1.5-1.8B', name: 'Qwen1.5 1.8B' },
          { id: 'Qwen/Qwen1.5-4B', name: 'Qwen1.5 4B' },
          { id: 'Qwen/Qwen1.5-7B', name: 'Qwen1.5 7B' },
        ]
      },
      'gemma': {
        name: 'Gemma',
        versions: [
          { id: 'google/gemma-2b', name: 'Gemma 2B' },
          { id: 'google/gemma-7b', name: 'Gemma 7B' },
          { id: 'google/gemma-2b-it', name: 'Gemma 2B Instruct' },
          { id: 'google/gemma-7b-it', name: 'Gemma 7B Instruct' },
        ]
      }
    };

    const handleModelChange = (e) => {
      const modelFamily = e.target.value;
      setSelectedModel(modelFamily);
      setSelectedVersion('');
      
      // Set default model name based on selection
      if (modelFamily && modelOptions[modelFamily]) {
        const defaultVersion = modelOptions[modelFamily].versions[0];
        setModelName(`Custom ${modelOptions[modelFamily].name}`);
      }
    };

    const handleSubmit = (e) => {
      e.preventDefault();
      
      const modelData = {
        id: editModel ? editModel.id : Date.now(),
        name: modelName || `Custom ${modelOptions[selectedModel]?.name || 'Model'}`,
        baseModel: modelOptions[selectedModel]?.versions.find(v => v.id === selectedVersion)?.name || '',
        status: editModel ? editModel.status : 'ready',
        settings: {
          quantization,
          device,
          cloudProvider,
          modelId: selectedVersion
        }
      };
      
      // Call the appropriate function based on mode
      if (onAddModel) {
        onAddModel(modelData);
      }
      
      // Close the modal
      onClose();
    };

    if (!isOpen) return null;

    return (
              <div className="modal-overlay" onClick={onClose}>
        <div className="modal" onClick={(e) => e.stopPropagation()}>
          <div className="modal-header">
            <h2>{editModel ? 'Edit LLM' : 'Add New LLM'}</h2>
            <button className="modal-close" onClick={onClose}>×</button>
          </div>
          
          <form onSubmit={handleSubmit}>
            <div className="modal-body">
              <div className="form-group">
                <label>Model Family</label>
                <select value={selectedModel} onChange={handleModelChange} required>
                  <option value="">Select a model family</option>
                  {Object.entries(modelOptions).map(([key, value]) => (
                    <option key={key} value={key}>{value.name}</option>
                  ))}
                </select>
              </div>

              {selectedModel && (
                <div className="form-group">
                  <label>Model Version</label>
                  <select value={selectedVersion} onChange={(e) => setSelectedVersion(e.target.value)} required>
                    <option value="">Select a version</option>
                    {modelOptions[selectedModel].versions.map((version) => (
                      <option key={version.id} value={version.id}>{version.name}</option>
                    ))}
                  </select>
                </div>
              )}

              <div className="form-group">
                <label>Model Name</label>
                <input
                  type="text"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                  placeholder="Give your model a name"
                  required
                />
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label>Quantization</label>
                  <select value={quantization} onChange={(e) => setQuantization(e.target.value)}>
                    <option value="none">None (Full Precision)</option>
                    <option value="8bit">8-bit</option>
                    <option value="4bit">4-bit</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Device</label>
                  <select value={device} onChange={(e) => setDevice(e.target.value)}>
                    <option value="cuda">GPU (CUDA)</option>
                    <option value="cpu">CPU</option>
                    <option value="mps">Apple Silicon (MPS)</option>
                  </select>
                </div>
              </div>

              <div className="form-group">
                <label>
                  Compute Location <Info size={14} className="inline-icon" />
                </label>
                <select value={cloudProvider} onChange={(e) => setCloudProvider(e.target.value)}>
                  <option value="local">Local</option>
                  <option value="lambda">Lambda Labs</option>
                  <option value="sagemaker">AWS SageMaker</option>
                  <option value="colab">Google Colab</option>
                </select>
                <p className="help-text">Choose where to run your model. Local requires sufficient hardware, cloud options handle the compute for you.</p>
              </div>
            </div>

            <div className="modal-footer">
              <button type="button" className="secondary-btn" onClick={onClose}>
                Cancel
              </button>
              <button type="submit" className="primary-btn">
                {editModel ? 'Save Changes' : 'Add Model'}
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  };
  
  return (
    <div className="app">
      <div className="container">
        <div className="header">
          <h1>Quintesson</h1>
          <p>Your LLMs</p>
        </div>
        
        <div className="models-section">
          <div className="models-grid">
            {models.map((model) => (
              <ModelCard
                key={model.id}
                model={model}
                isSelected={selectedModel?.id === model.id}
                onClick={setSelectedModel}
                onEdit={openEditModal}
              />
            ))}
          </div>
          
          <button className="new-llm-btn" onClick={() => setShowNewLLMModal(true)}>
            <Plus size={16} />
            New LLM
          </button>
        </div>
        
        {selectedModel && (
          <div className="edit-section">
            <h2>Edit "{selectedModel.name}"</h2>
            
            <div className="tabs">
              <nav className="tabs-nav">
                {modelTabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`tab ${activeTab === tab.id ? 'active' : ''}`}
                  >
                    <div className="tab-header">
                      <Info size={16} />
                      <span>{tab.title}</span>
                    </div>
                    <p>{tab.description}</p>
                  </button>
                ))}
              </nav>
            </div>
            
            <TabContent tab={modelTabs.find(t => t.id === activeTab)} />
          </div>
        )}
      </div>

      {/* Modals */}
      <NewLLMModal 
        isOpen={showNewLLMModal} 
        onClose={() => setShowNewLLMModal(false)}
        onAddModel={handleAddModel}
      />
      
      <NewLLMModal 
        isOpen={showEditModal} 
        onClose={() => {
          setShowEditModal(false);
          setModelToEdit(null);
        }}
        onAddModel={handleEditModel}
        editModel={modelToEdit}
      />
    </div>
  );
};

export default PeftApp;