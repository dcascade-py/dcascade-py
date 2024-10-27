%seven is fractions
%9 is reaches

dir('C:\bin\cascade\Oct25RangitataFC_dH')

%casc_result_dir = 'C:\bin\cascade\Rangitata_Rev10_Qsed_10pct_WC_updateD84_gorge0'
casc_result_dir = 'C:\bin\cascade\Oct25RangitataFC_dH\Rev2_5pctsand_1234'
%casc_result_dir = 'C:\bin\cascade\RangitataFC_dH\Rev1_noQsed_GorgeHydrGeo'

%casc_result_dir = 'C:\bin\cascade\Rangitata_Rev8_Qsed_10pct_10xsandcap'

dir(casc_result_dir)

inQs = dlmread('C:\bin\cascade\RangitataFC_dH\qsand_40pct_gravUpper68_2024.csv');

Qindex = 5 %reach where I measure Q for histograms. set to the gorge (Klondyke gauge). Could also set to just below but I think we stay
%in the Q-at-gauge system. 
%%

for nrepeats = 0:1
C = load([ casc_result_dir filesep 'save_all_' num2str(nrepeats) '.mat'])
%C = load('C:\bin\cascade\Rangitata_Rev3_results\ConstWidth_save_all_10.mat')
if nrepeats == 0,linew=2,else,linew=1,end

% Move fields from C.data_output to C
tmaxindex = length(C.data_output.V_dep_sum); %some go to timescale, some to timescale-1 in python output code. let's cut off n=timescale
fields = fieldnames(C.data_output);  % Get all the field names from data_output
for i = 1:numel(fields)
    C.(fields{i}) = C.data_output.(fields{i});  % Assign each field to the main structure
end

fields = fieldnames(C.extended_output);  % Get all the field names from data_output
for i = 1:numel(fields)
    C.(fields{i}) = C.extended_output.(fields{i});  % Assign each field to the main structure
end

% Remove the data_output substructure
C = rmfield(C, 'data_output');C = rmfield(C, 'extended_output');


Qbry = C.Q(:,Qindex);Qbry(end)=NaN;
dmi = C.dmi;
psi = C.psi;
Tdays_rep=C.Tdays(1:end-1) + nrepeats*C.Tdays(end-1) ;
issand = dmi<2;

n_reaches = size(C.slope,2)
n_class = length(psi)

n_reach = 5
figure(2), %clf
plot(Tdays_rep,C.flow_depth(1:end-1,n_reach),'k--'), hold on
plot(Tdays_rep,C.Q(1:end-1,n_reach) ./ (C.flow_depth(1:end-1,n_reach) .* C.wac(1:end-1,n_reach)),'g--')

%V-sed is a virtual velocity, that is a sediment load divided by an active area. 
%if sand's V-sed is low, I probably need to increase capacity, not mess with V_sed calculation.
%is it finally time to bring in my VR conc lim stuff somewhere that can add??
for n_reach = 1
plot(Tdays_rep,C.V_sed(1:end-1,n_reach,7),'c--'),ylabel('m/dt or [m]'), hold on
plot(Tdays_rep,C.V_sed(1:end-1,n_reach,6),'c-'),ylabel('m/dt or [m]'), hold on
plot(Tdays_rep,C.V_sed(1:end-1,n_reach,2),'b'),ylabel('m/dt or [m]'), hold on
end
legend('Depth','U','V_Sand 125','V_Sand 500','V_Cobble')
title('Sed Velocity')

size(C.V_sed(:,:,n_class))

C.Fs = sum(C.Fi_r_ac(:,:,6:7),3); %

%ylim([0 4.5]), grid on

reachlen = cumsum(C.Length)
sk=24
%%

%need to aggregate transport into Q-histograms, and ideally annualize. 
dq=50
qranges = [0:dq:1200]'; %same as qedges
qranges_top = [dq:dq:1200]';
nqbins = length(qranges_top)
[qN qedges qbin] = histcounts(C.Q(:,Qindex),qranges); %use qbin to sum by sediment loads

%Qwsum = accumarray(qbin,C.Q(:,1)) %what size is this returning :( 
trcapsum = zeros(nqbins,n_reaches,n_class);
transp_class_sum = trcapsum; dep_class_sum = trcapsum; mobil_class_sum = trcapsum;
for q=1:nqbins
    inds = C.Q(1:end-1,5) >= qranges(q) &  C.Q(1:end-1,5) < qranges(q+1); %times when q in qbin
    trcapsum(q,:,:) = squeeze(sum(C.tr_cap_class(:,inds,:),2))' ; %sum sediment capacity across time dim only

    transp_class_sum(q,:,:)= squeeze(sum(C.transported_class(:,inds,:),2))' ; %sum sediment capacity across time dim only
    %this is ALL deposited sed. so the volume of whole stratigraphy. not the deposition FLUX
    %dep_class_sum(q,:,:)   = squeeze(sum(C.deposited_class(:,inds,:),2))' ; %sum sediment capacity across time dim only
    %This is the change in deposited volume, which is what we want to compare to the others. awkward dim change.
    dep_class_sum(q,:,:)   = squeeze(sum(C.Delta_V_class(inds,:,:),1)) ; %sum sediment capacity across time dim only
    
    mobil_class_sum(q,:,:) = squeeze(sum(C.mobilised_class(:,inds,:),2))' ; %sum sediment capacity across time dim only
    %sum deposition 

    %sum erosion
,end
figure(110),clf
subplot(311)
plot(qsteps,histcounts(C.Q(:,Qindex),[0 qsteps])), hold on
title("Q distribution of run")

reachestosum = 6:14;
%first dim: q  second dim reach. third dim - grain size
sandcapsum = sum(trcapsum(:,reachestosum,issand),3);gravcapsum = sum(trcapsum(:,reachestosum,~issand),3);
sandtranspsum = sum(transp_class_sum(:,reachestosum,issand),3);gravtranspsum = sum(transp_class_sum(:,reachestosum,~issand),3);
sanddepsum = sum(dep_class_sum(:,reachestosum,issand),3);gravdepsum = sum(dep_class_sum(:,reachestosum,~issand),3);
sandmobsum = sum(mobil_class_sum(:,reachestosum,issand),3);gravmobsum = sum(mobil_class_sum(:,reachestosum,~issand),3);
subplot(312)

area(qranges_top,[gravmobsum(:,1)  sandmobsum(:,1) ])
hold on %what sign is deposition?
area(qranges_top,-[gravdepsum(:,1) sanddepsum(:,1)  ])
%area(qranges_top,[gravtranspsum(:,2) sandtranspsum(:,2) ]) %nothing transported INTO 1. 

legend('gravel cap','sand cap','gravel dep','sand dep')
xlabel('Q bin'),ylabel('capacity, sum of m3/hour')

sandcaplongsum = sum(trcapsum(:,:,issand),[1 3]); gravcaplongsum = sum(trcapsum(:,:,~issand),[1 3]);
sandtransplongsum = sum(transp_class_sum(:,:,issand),[1 3]); gravtransplongsum = sum(transp_class_sum(:,:,~issand),[1 3]);
sanddeplongsum = sum(dep_class_sum(:,:,issand),[1 3]); gravdeplongsum = sum(dep_class_sum(:,:,~issand),[1 3]);
sandmoblongsum = sum(mobil_class_sum(:,:,issand),[1 3]); gravmoblongsum = sum(mobil_class_sum(:,:,~issand),[1 3]);
hold on

subplot(313)
area(reachlen,[gravcaplongsum' sandcaplongsum' ])
hold on
bar(reachlen,[gravtransplongsum'  sandtransplongsum' ],'stacked','facealpha',.2)
bar(reachlen,[gravmoblongsum' sandmoblongsum'  ],.4,'stacked','facealpha',.4)
bar(reachlen,-[gravdeplongsum' sanddeplongsum'  ],.8,'stacked','facealpha',.8)
%%
figure(11), 
%Qbi_tr = [np.zeros((n_reaches,n_reaches,n_classes), dtype=numpy.float32) for _ in range(timescale)] 
% %# sediment within the reach AFTER transfer, which also gives the provenance 
[qmax ti] = max(Qbry)
ti=ti-1
%Qbi_mob = [np.zeros((n_reaches,n_reaches,n_classes), dtype=numpy.float32) for _ in range(timescale)] 
% %# sediment within the reach BEFORE transfer, which also gives the provenance 
subplot(3,3,1)

%elevation
plot([0 reachlen],C.Node_el(1,1:end),'k-','linew',linew)
hold on
plot([0 reachlen],C.Node_el(end,1:end),'r-','linew',linew)

legend('el start','el at end')

subplot(3,3,2)
%slope pcolor

pcolor(1:n_reaches,Tdays_rep(1:24:end-25),diff(C.slope(1:24:end-1,:)))
shading flat
caxis([-1e-5 1e-5]),colormap(ttscm('vik'));,colorbar off
hold on, ylim([0 max(Tdays_rep)])

subplot(3,3,3)
%D50
h31(1) = plot(reachlen/1e3,1000*C.D50_AL([1],:),'k--','linew',linew), hold on
h31(2) = plot(reachlen/1e3,1000*C.D50_AL([end-1],:),'r--','linew',linew), hold on
h32(1) = plot(reachlen/1e3,100*C.Fs([1],:),'m-','linew',linew)
h32(2) = plot(reachlen/1e3,100*C.Fs([end-1],:),'m--','linew',linew)
%h33 = plot(reachlen/1e3,1000*C.D50_mob([1 end-1],:),'k-','linew',linew)
h34(1) = plot(reachlen/1e3,1000*C.D50_dep([1],:),'g:','linew',linew)
h34(2) = plot(reachlen/1e3,1000*C.D50_dep([end-1],:),'g:','linew',linew)
xlabel('[Downstream Distance [km]')
legend('D50 AL 1','D50 AL end','Fs 1', 'Fs end', 'dep 1','dep end')
subplot(3,3,4)
%hydraulics at a few q levels

[foo, qsi] = sort(Qbry) %sorted indices
qperclist = [10 50 90 95 98]
qilist = []; %save indices to flows at the above percentiles
for qq =1:length(qperclist)  %get a few longitudinal depths/velocities at flows of interest
    qperc = qperclist(qq)
    Q_at_qperc = prctile(Qbry,qperc)
     
    qi = qsi(find(foo>=Q_at_qperc,1,'first'));
    qilist = [qilist; qi]
    Qbry(qi)
    plot(reachlen,C.flow_depth(qi,:),'k-'), hold on
    plot(reachlen,C.Q(qi,:)./(C.flow_depth(qi,:) .* C.wac(qi,:)),'r-')
    ylim([0 2.5]), grid on
end    
legend('h','U')



%plot h and v
 

subplot(3,3,5)
%longitudinal deposition, erosion, transported plots

t=ti
plot(reachlen/1000,squeeze(sum(C.Qbi_tr(qilist,:,:,nsed),[2])),'r','linew',linew), hold on
plot(reachlen/1000,squeeze(sum(C.Qbi_mob(qilist,:,:,nsed),[2])),'b','linew',linew)
plot(reachlen/1000,squeeze(C.Q_out(qilist,:,nsed)),'k','linew',linew)
title('sum across FIRST dim of nreaches - should sum this and uptsream contributions')
xlabel('Downstream Distance [km]')
ylabel('transport, m3/hour, one flow')

subplot(3,3,6)
%GSD of reaches, sources in ECDF form
imagesc(reachlen/1000,Tdays_rep(1:24:end-25),100*C.Fs(1:24:end-25,:))
shading flat,axis xy, hold on
caxis([0 100]),colormap(ttscm('vik'));,colorbar off
title('Fs'), hold on,ylabel("Days")
ylim([0 max(Tdays_rep)])
subplot(3,3,7) 
%compare 1D hydraulics to 2D

subplot(3,3,8)
%stacked area plot of t/year transport vs inputs

subplot(3,3,9)
%GSDs starting, boundaries, end

 Perc_finer_ac=zeros(n_reaches,n_class);
 Perc_finer_ac(:,1)=100;
 for t=[1 length(C.Tdays)-1]
    for i = 2:size(Perc_finer_ac,2)  %dmi
        for n=1:size(Perc_finer_ac,1)  %NR
            Perc_finer_ac(n,i)=Perc_finer_ac(n,i-1) - (C.Fi_r_ac(t,n,i-1)*100);
        end
    end
    C.Perc_finder_ac(t,1:n_reaches,1:n_class) = Perc_finer_ac;
 end
 %plot every other reach GSD in cumulative form. 
    plot(psi,squeeze(C.Perc_finder_ac(1,1:2:end,:)),'k--','linew',linew), hold on
    plot(psi,squeeze(C.Perc_finder_ac(length(C.Tdays)-1,1:2:end,:)),'r:')
    set(gca,'xdir','reverse')
    set(gca,'xtick',psi(2:2:end),'xticklabel',xStrings), xlim(sed_range),
    %%

    CASC_py_diagnose_transport

end


%%
% 
% figure(12)
% nreach = 2
% nsed = 7
% plot(Qbry,sum(C.Qbi_mob(:,:,nreach,nsed),[2 3]),'k-','linew',linew), hold on
% plot(Qbry,sum(C.Qbi_mob(:,:,nreach+1,nsed),[2 3]),'g','linew',linew)
% plot(Qbry,sum(C.Qbi_mob(:,:,nreach-1,nsed),[2 3]),'r','linew',linew)
% title('Q vs Qbi_mob')
% legend(['Q vs mob, reach' num2str(nreach)],['Q vs mob, reach' num2str(nreach+1)],['Q vs mob, reach' num2str(nreach-1)] )
% 
% %  

%%
%sand frac investigations
% figure(501),plot(Qbry,C.Fs,'.')
% title('Fs rating curve?')
% 
% figure(502), %clf,
% plot(Tdays_rep,100*C.Fs(1:end-1,1),'k'), hold on
% plot(Tdays_rep,100*C.Fs(1:end-1,2),'b')
% 
% plot(Tdays_rep,100*C.Fs(1:end-1,2:2:9),'b--')
% 
% ylabel('Fs %')
% title('Fs timeseries?')
