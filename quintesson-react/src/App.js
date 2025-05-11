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
      status: 'ready'
    }
  ]);

  const [selectedModel, setSelectedModel] = useState(models[0]);
  const [activeTab, setActiveTab] = useState('peft');

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
      description: 'Test your LLM for safety concerns'
    }
  ];

  const ModelCard = ({ model, isSelected, onClick }) => (
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
          <button className="edit-btn">
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
              />
            ))}
          </div>
          
          <button className="new-llm-btn">
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
    </div>
  );
};

export default PeftApp;