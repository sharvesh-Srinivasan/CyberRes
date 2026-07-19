import { useState, useEffect } from 'react'
import ForceGraph2D from 'react-force-graph-2d'

export default function App() {
  const [status, setStatus] = useState<string>('initializing')
  const [metrics, setMetrics] = useState<any>(null)
  const [entities, setEntities] = useState<any[]>([])
  const [graphData, setGraphData] = useState<any>(null)
  const [auditLogs, setAuditLogs] = useState<any[]>([])
  const [activeTab, setActiveTab] = useState<string>('overview')
  const [lightMode, setLightMode] = useState<boolean>(false)

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/status')
      const data = await res.json()
      setStatus(data.status)
      if (data.status === 'ready') {
        fetchMetrics()
        fetchEntities()
        fetchGraph()
        fetchAudit()
      }
    } catch (e) {
      console.error(e)
    }
  }

  const fetchMetrics = async () => {
    try {
      const res = await fetch('/api/metrics')
      if (res.ok) {
        const data = await res.json()
        setMetrics(data)
      }
    } catch (e) { console.error(e) }
  }

  const fetchEntities = async () => {
    try {
      const res = await fetch('/api/entities')
      if (res.ok) {
        const data = await res.json()
        setEntities(data.entities)
      }
    } catch (e) { console.error(e) }
  }

  const fetchGraph = async () => {
    try {
      const res = await fetch('/api/graph')
      if (res.ok) {
        const data = await res.json()
        setGraphData(data)
      }
    } catch (e) { console.error(e) }
  }

  const fetchAudit = async () => {
    try {
      const res = await fetch('/api/audit')
      if (res.ok) {
        const data = await res.json()
        setAuditLogs(data.audit_logs)
      }
    } catch (e) { console.error(e) }
  }

  const handleOverride = async (entityId: string, action: string) => {
    try {
      const res = await fetch('/api/override', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ entity_id: entityId, action })
      })
      if (res.ok) {
        fetchEntities()
        fetchAudit()
      }
    } catch (e) { console.error(e) }
  }

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 5000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (lightMode) {
      document.body.classList.add('light-theme')
    } else {
      document.body.classList.remove('light-theme')
    }
  }, [lightMode])

  if (status === 'initializing' || !metrics) {
    return (
      <div style={{ display: 'flex', height: '100vh', alignItems: 'center', justifyContent: 'center', backgroundColor: 'var(--bg-app)' }}>
        <h1 className="mono blinking-cursor" style={{ fontSize: '16px', color: 'var(--text-secondary)' }}>
          SYSTEM INITIALIZING... PLEASE STAND BY
        </h1>
      </div>
    )
  }

  return (
    <div className="app-layout">
      
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            </svg>
            Cyber Command
          </div>
        </div>
        <nav className="sidebar-nav">
          <button className={`nav-item ${activeTab === 'overview' ? 'active' : ''}`} onClick={() => setActiveTab('overview')}>
            <svg className="nav-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>
            Overview
          </button>
          <button className={`nav-item ${activeTab === 'graph' ? 'active' : ''}`} onClick={() => setActiveTab('graph')}>
            <svg className="nav-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="18" cy="5" r="3"></circle><circle cx="6" cy="12" r="3"></circle><circle cx="18" cy="19" r="3"></circle><line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line><line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line></svg>
            Network Graph
          </button>
          <button className={`nav-item ${activeTab === 'audit' ? 'active' : ''}`} onClick={() => setActiveTab('audit')}>
            <svg className="nav-item-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
            Audit Log
          </button>
        </nav>
      </aside>

      {/* Main Content Area */}
      <div className="main-area">
        <header className="topbar">
          <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
            <span className="badge badge-success">System Secure</span>
            <button className="btn btn-secondary btn-sm" onClick={() => setLightMode(!lightMode)}>
              {lightMode ? 'Dark Mode' : 'Light Mode'}
            </button>
          </div>
        </header>

        <main className="content-container">
          
          {activeTab === 'overview' && (
            <>
              <div className="page-header">
                <h2 className="page-title">Platform Overview</h2>
              </div>

              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-label">False Positive Rate</div>
                  <div className="metric-value text-success">{(metrics.fpr * 100).toFixed(2)}%</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Recall Rate</div>
                  <div className="metric-value text-success">{(metrics.recall * 100).toFixed(2)}%</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Samples Evaluated</div>
                  <div className="metric-value">{metrics.n_samples.toLocaleString()}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-label">Average Latency</div>
                  <div className="metric-value text-warning">{metrics.latency_ms.toFixed(0)} ms</div>
                </div>
              </div>

              <div className="card">
                <div className="card-header">
                  Flagged Entities & BFT Consensus
                </div>
                <div className="card-body" style={{ padding: 0 }}>
                  <table>
                    <thead>
                      <tr>
                        <th>Entity ID</th>
                        <th>Risk Score</th>
                        <th>Category</th>
                        <th>Consensus Votes</th>
                        <th>Action Applied</th>
                        <th>Manual Override</th>
                      </tr>
                    </thead>
                    <tbody>
                      {entities.slice(0, 15).map((ent, i) => (
                        <tr key={i}>
                          <td className="mono">
                            {ent.id} 
                            {ent.tier0 && <span className="badge badge-warning" style={{ marginLeft: '8px' }}>Tier 0</span>}
                          </td>
                          <td className="mono text-error">{ent.risk_score.toFixed(3)}</td>
                          <td><span className="badge badge-neutral">{ent.category}</span></td>
                          <td>
                            <div style={{ display: 'flex', gap: '4px' }}>
                              {ent.votes && ent.votes.map((v: boolean, vi: number) => (
                                <div key={vi} style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: v ? 'var(--status-error-text)' : 'var(--status-success-text)' }} title={v ? 'Flagged' : 'Clean'} />
                              ))}
                            </div>
                          </td>
                          <td>
                            <span className={`badge ${ent.decision === 'Auto-Contain' ? 'badge-error' : 'badge-warning'}`}>
                              {ent.decision}
                            </span>
                          </td>
                          <td>
                            <div style={{ display: 'flex', gap: '8px' }}>
                              <button className="btn btn-primary btn-sm" onClick={() => handleOverride(ent.id, 'Escalate')}>Escalate</button>
                              <button className="btn btn-secondary btn-sm" onClick={() => handleOverride(ent.id, 'Dismiss')}>Dismiss</button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}

          {activeTab === 'graph' && (
            <>
              <div className="page-header">
                <h2 className="page-title">Network Intelligence Graph</h2>
              </div>
              <div className="card" style={{ height: 'calc(100vh - 220px)', padding: 0 }}>
                {graphData ? (
                  <ForceGraph2D
                    graphData={graphData}
                    nodeAutoColorBy="group"
                    nodeLabel="id"
                    backgroundColor={lightMode ? '#ffffff' : '#18181b'}
                  />
                ) : (
                  <div style={{ padding: '32px', color: 'var(--text-secondary)' }}>Loading network graph data...</div>
                )}
              </div>
            </>
          )}

          {activeTab === 'audit' && (
            <>
              <div className="page-header">
                <h2 className="page-title">System Audit Log</h2>
              </div>
              <div className="card">
                <div className="card-header">
                  Immutable SHA-256 Records
                </div>
                <div className="card-body" style={{ padding: 0 }}>
                  <table>
                    <thead>
                      <tr>
                        <th>Timestamp (UTC)</th>
                        <th>Entity ID</th>
                        <th>System Decision</th>
                        <th>SHA-256 Hash Record</th>
                      </tr>
                    </thead>
                    <tbody>
                      {auditLogs.map((log, i) => (
                        <tr key={i}>
                          <td className="mono text-muted">{new Date(log.timestamp * 1000).toISOString().replace('T', ' ').substring(0, 19)}</td>
                          <td className="mono">{log.entity_id}</td>
                          <td>
                            <span className={`badge ${log.decision === 'Auto-Contain' ? 'badge-error' : 'badge-warning'}`}>
                              {log.decision} {log.override_applied && ' (Overridden)'}
                            </span>
                          </td>
                          <td className="mono text-muted" style={{ fontSize: '12px' }}>{log.hash}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
          
        </main>
      </div>
    </div>
  )
}
