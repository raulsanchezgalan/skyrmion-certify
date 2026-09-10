"""Regenerate the geometry schematic and two figures of numerical results."""
from pathlib import Path
import json,logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon
from physical_model import Flake
logging.getLogger('fontTools.ttLib.tables._h_e_a_d').setLevel(logging.ERROR)

HERE=Path(__file__).resolve().parent;DATA=HERE/'data';FIG=HERE.parent/'figures'
BLUE='#24618d';ORANGE='#b65b21';GREEN='#47745a'
plt.rcParams.update({'font.family':'serif','font.size':10,'mathtext.fontset':'cm',
 'axes.labelsize':11,'axes.titlesize':11,'legend.fontsize':9,'xtick.labelsize':9,'ytick.labelsize':9,
 'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
 'lines.linewidth':1.5,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':200})

def label(ax,s):ax.text(-.15,1.03,s,transform=ax.transAxes,fontsize=11,fontweight='bold')
def load(name):return json.load(open(DATA/(name+'.json')))
def phase_plot(ax,rows,key,color,label_text,marker=None):
    q=np.rint([r['Q'] for r in rows]).astype(int);cuts=np.r_[0,np.where(q[1:]!=q[:-1])[0]+1,len(rows)]
    for a,b in zip(cuts[:-1],cuts[1:]):
        ax.plot([r['B'] for r in rows[a:b]],[r[key] for r in rows[a:b]],color=color,
                marker=marker,markersize=3.5,label=label_text if a==0 else None)

def texture(ax,flake,m,title):
    pos=flake.mesh_positions[:-1];ext=flake.extend(m)[:-1]
    edges=set()
    for tri in flake.faces:
        if max(tri)>=len(pos):continue
        for i,j in zip(tri,tri[1:]+tri[:1]):edges.add(tuple(sorted((i,j))))
    for i,j in edges:ax.plot(pos[[i,j],0],pos[[i,j],1],color='#d4d6d8',lw=.55,zorder=0)
    norm=Normalize(-1,1);cmap=plt.get_cmap('RdBu_r');N=flake.N
    ax.scatter(pos[N:,0],pos[N:,1],c=2*ext[N:,2],cmap=cmap,norm=norm,marker='s',s=43,edgecolor='#666666',linewidth=.5,zorder=2)
    ax.scatter(pos[:N,0],pos[:N,1],c=2*m[:,2],cmap=cmap,norm=norm,s=85,edgecolor='#252525',linewidth=.6,zorder=3)
    ax.quiver(pos[:N,0],pos[:N,1],m[:,0],m[:,1],angles='xy',scale_units='xy',scale=.60,
              color='#202020',width=.008,headwidth=3.8,headlength=4.8,zorder=4,pivot='tail')
    ax.set_aspect('equal');ax.set_title(title);ax.set_xlabel('$x/a$');ax.set_ylabel('$y/a$')
    bound=flake.radius+1.35;ax.set_xlim(-bound,bound);ax.set_ylim(-bound,bound)
    return ScalarMappable(norm=norm,cmap=cmap)

def admissibility_schematic():
    """Exact planar example of the equal-weight projection construction.

    Coordinates are schematic moment data, not results of a spin simulation.
    All norms are below 1/2; q=(.22,0) is on the vertical supporting edge.
    """
    vertices=np.array([[.22,-.23],[.22,.24],[.43,.07]])
    q=np.array([.22,0.])
    assert np.max(np.linalg.norm(vertices,axis=1))<.5
    assert np.min((vertices-q)@q)>-1e-15
    fig,axes=plt.subplots(1,2,figsize=(7.05,2.65))
    fig.subplots_adjust(left=.025,right=.995,bottom=.03,top=.90,wspace=.16)
    for k,ax in enumerate(axes):
        m=vertices if k==0 else vertices-q
        ax.add_patch(Polygon(m,closed=True,facecolor='#dbe8ee',edgecolor=BLUE,lw=1.4))
        ax.scatter(m[:,0],m[:,1],s=20,c=BLUE,zorder=4)
        ax.scatter([0],[0],s=27,c='#151515',zorder=6)
        for a,point,offset in zip('ijk',m,[(-.01,-.052),(-.01,.027),(.025,0.)]):
            prime="'" if k else ''
            ax.text(point[0]+offset[0],point[1]+offset[1],
                    '$m_'+a+prime+'$',fontsize=12,ha='center',va='center')
        ax.set_aspect('equal');ax.set_xlim(-.09,.55);ax.set_ylim(-.39,.34);ax.axis('off')
        ax.set_title('(a) Admissible face' if k==0 else '(b) Attaining perturbation',loc='left',pad=5)
    ax=axes[0]
    ax.plot([q[0],q[0]],[-.32,.31],ls='--',lw=.8,color='#717171',zorder=0)
    ax.plot([0,q[0]],[0,0],lw=1.,color='#303030')
    ax.scatter([q[0]],[0],s=29,c=ORANGE,zorder=6)
    ax.text(-.032,-.042,'$0$',fontsize=12)
    ax.text(q[0]+.017,-.035,'$q$',fontsize=12,color=ORANGE)
    ax.text(.30,.03,'$P_\\tau$',fontsize=12,color=BLUE)
    ax.annotate('',xy=(.135,0),xytext=(0,0),arrowprops={'arrowstyle':'-|>','lw':1.5,'color':GREEN},zorder=7)
    ax.text(.055,.036,'$v$',fontsize=12,color=GREEN)
    ax.annotate('',xy=(q[0],-.33),xytext=(0,-.33),arrowprops={'arrowstyle':'<->','lw':.9,'color':'#404040'})
    ax.text(q[0]/2,-.377,r'$r_\tau=|q|$',fontsize=11,ha='center')
    ax=axes[1]
    ax.text(-.033,-.045,'$0$',fontsize=12)
    ax.annotate('$P_\\tau-q$',xy=(.075,.045),xytext=(.25,.19),
                fontsize=12,color=BLUE,ha='center',
                arrowprops={'arrowstyle':'-','lw':.7,'color':BLUE})
    ax.text(.19,-.33,r'$m_r^{\prime}=m_r-q$',fontsize=11,ha='center')
    for ext in ['pdf','png']:fig.savefig(FIG/f'admissibility_geometry.{ext}')
    plt.close(fig)

def make():
    FIG.mkdir(exist_ok=True)
    admissibility_schematic()
    small=load('n7_scan');large=load('n19_scan');s=np.load(DATA/'n7_scan.npz');l=np.load(DATA/'n19_scan.npz')
    fig=plt.figure(figsize=(7.05,6.25),layout='constrained')
    gs=fig.add_gridspec(2,2,height_ratios=[1.12,1.0]);axes=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
    si=int(np.argmin(abs(s['B']-.2)));li=next(i for i,r in enumerate(large) if r['B']==.2)
    sm=texture(axes[0],Flake(1),s['moments'][si],'$N=7$, $B=0.2J$')
    texture(axes[1],Flake(2),l['moments'][li],'$N=19$, $B=0.2J$')
    cb=fig.colorbar(sm,ax=axes[:2],fraction=.037,pad=.035,shrink=.8);cb.set_label('$2m_z$');cb.set_ticks([-1,0,1])
    phase_plot(axes[2],small,'gap',BLUE,'$N=7$')
    phase_plot(axes[2],large,'gap',ORANGE,'$N=19$','o')
    phase_plot(axes[3],small,'radius_B',BLUE,'$N=7$')
    phase_plot(axes[3],large,'radius_B',ORANGE,'$N=19$','o')
    axes[2].set_ylabel(r'$\Delta/J$');axes[3].set_ylabel('$R_B/J$')
    for ax in axes[2:]:ax.set_xlabel('$B/J$');ax.set_xlim(0,2);ax.set_ylim(bottom=0);ax.grid(alpha=.18);ax.legend(frameon=False,loc='upper left')
    ref=small[si];axes[3].plot(.2,ref['radius_B'],'D',color='#191919',ms=4,zorder=5)
    axes[3].annotate('verified reference',xy=(.2,ref['radius_B']),xytext=(.4,.135),fontsize=8,
                     arrowprops={'arrowstyle':'-','lw':.7,'color':'#444444'})
    for ax,t in zip(axes,['(a)','(b)','(c)','(d)']):label(ax,t)
    for ext in ['pdf','png']:fig.savefig(FIG/f'flakes_and_bounds.{ext}')
    plt.close(fig)

    rows=load('n7_transverse');event=load('geometric_event')['row'];event['Gamma']=0.;event['radius_B']=0.
    rows=sorted(rows+[event],key=lambda r:r['B']);B=np.array([r['B'] for r in rows]);Bstar=event['B']
    fig,axes=plt.subplots(2,2,figsize=(7.05,5.1),layout='constrained',sharex=True)
    for ax,key,ylabel,t in zip(axes.flat,['gap','Gamma','p_min','radius_B'],
                              [r'$\Delta/J$',r'$\Gamma$',r'$p_{\min}$','$R_B/J$'],['(a)','(b)','(c)','(d)']):
        ax.axvspan(.648,.649,color='#d8c48a',alpha=.48,zorder=0)
        ax.axvline(Bstar,color='#555555',ls='--',lw=.9)
        ax.plot(B,[r[key] for r in rows],color=BLUE)
        ax.set_xlim(.62,.68);ax.set_ylabel(ylabel);ax.grid(alpha=.15);label(ax,t)
        if key!='p_min':ax.set_ylim(bottom=0)
        if key=='p_min':ax.set_ylim(.3,1)
    axes[0,1].text(.07,.90,'$Q=-1$',transform=axes[0,1].transAxes)
    axes[0,1].text(.77,.90,'$Q=0$',transform=axes[0,1].transAxes)
    for ax in axes[1]:ax.set_xlabel('$B/J$')
    for ext in ['pdf','png']:fig.savefig(FIG/f'gapped_charge_change.{ext}')
    plt.close(fig)
    return [str(FIG/'admissibility_geometry.pdf'),str(FIG/'flakes_and_bounds.pdf'),str(FIG/'gapped_charge_change.pdf')]

if __name__=='__main__':print('\n'.join(make()))
